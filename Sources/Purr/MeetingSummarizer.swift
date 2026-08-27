import Foundation
import os.log

#if canImport(FoundationModels) && !DISABLE_AFM
import FoundationModels
#endif

// Generates Minutes of Meeting from a transcript Markdown file.
// Output is a sidecar "<transcript>.summary.md" so the transcript stays
// untouched and a re-run only rewrites the summary.
//
// Two backends, picked at call time:
//
//   * Apple Foundation Models (macOS 26+, Apple Intelligence enabled and
//     a supported language). Built-in, no download, ~3 B params,
//     ~30 tok/s. Preferred when available.
//   * Gemma 3 4B Instruct (Q4_K_M GGUF) via llama.cpp. ~2.49 GB on disk,
//     fetched from Hugging Face on first use. Used on macOS < 26, when
//     Apple Intelligence is unavailable, or when the user explicitly
//     picks it.
//
// The Apple Intelligence path takes no model-load cost. The Gemma path
// lazy-loads a single LlamaSession (reused across summaries) and runs
// with a bounded KV cache so very long meetings stay within memory.
//
// A single `MeetingSummarizer.shared` instance is used app-wide so
// download-then-summarize and Settings' delete button operate on the
// same loaded model.
@MainActor
final class MeetingSummarizer {
    static let shared = MeetingSummarizer()

    enum SummarizerError: LocalizedError {
        case modelLoadFailed(Error)
        case backendUnavailable
        case emptyResponse
        case unsupportedLocale
        case contentFlagged(String)
        case generationFailed(Error)

        var errorDescription: String? {
            switch self {
            case .modelLoadFailed(let underlying):
                return "Could not load summarization model: \(underlying.localizedDescription)"
            case .backendUnavailable: return "No summarization backend available."
            case .emptyResponse: return "Summarization model returned no text."
            case .unsupportedLocale:
                return
                    "Apple Foundation Model doesn't support this device's language. Switch to Gemma in Settings → Features."
            case .contentFlagged(let reason):
                return "On-device safety guardrails blocked the summary: \(reason)."
            case .generationFailed(let underlying):
                return
                    "Summarization failed: \(underlying.localizedDescription). The transcript was saved without a summary."
            }
        }
    }

    enum Backend: Equatable {
        case appleFoundation
        case llamaCpp(installed: Bool)
        case none
    }

    private let log = Logger(subsystem: "com.arunbrahma.purr", category: "summarizer")

    // Hard ceiling on transcript characters fed to the model. Longer
    // meetings get a head/tail truncation (the most useful content for a
    // summary lives at the start and the end). 24K chars ≈ 6K tokens,
    // which leaves room for the system prompt and a 1000-token response
    // inside the 4096-token context window.
    private static let maxTranscriptChars = 24_000

    // A transcript whose body has fewer words than this is treated as too
    // short to summarize. Below it there is nothing to minute, and a model
    // handed such a transcript fabricates a plausible meeting out of noise.
    // ~20 words is roughly 10 seconds of speech - shorter than any real
    // meeting worth a summary.
    private static let minWordsToSummarize = 20

    // Maximum wall-clock time we let the LLM produce a response.
    // Generous (5 min) so a hot run on a long meeting still finishes;
    // longer than that almost always means the model has wandered into a
    // repetition loop and we should bail.
    private static let generationTimeout: Duration = .seconds(300)

    private init() {}

    // Releases the in-memory LlamaSession owned by the shared runtime.
    // MUST run before deleting the on-disk weights — otherwise the ~2.5 GB
    // session stays resident and the next generation reuses a session
    // pointing at deleted files.
    func unload() async {
        log.info("Llama unload: releasing the shared runtime session.")
        await LlamaRuntime.shared.unload()
    }

    // MARK: - Backend availability

    // The chosen path for *this device right now*. Honours the user's
    // preference when both backends are usable; otherwise falls back
    // along a strict priority:
    //
    //   1. Apple FM (preferred when available - macOS 26+, AI on,
    //      locale supported via supportsLocale).
    //   2. llama.cpp + Gemma if downloaded.
    //   3. .none → caller refuses to summarize.
    //
    // We NEVER return .llamaCpp(installed: false) - the caller would
    // then trigger a silent multi-GB download from a meeting-stop code
    // path, which is exactly the surprise we refuse to ship.
    static func currentBackend() -> Backend {
        let installed = LLMModelManager.isInstalled()
        let appleAvailable: Bool = {
            if #available(macOS 26.0, *) { return appleFoundationAvailable }
            return false
        }()
        let preference = SettingsStore.shared.summaryBackend

        if preference == .llamaCpp, installed {
            return .llamaCpp(installed: true)
        }
        if appleAvailable {
            return .appleFoundation
        }
        if installed {
            return .llamaCpp(installed: true)
        }
        return .none
    }

    // True when at least one backend is ready to summarize without further user action.
    static var canSummarizeNow: Bool {
        switch currentBackend() {
        case .appleFoundation: return true
        case .llamaCpp(let installed): return installed
        case .none: return false
        }
    }

    // True when every gate Apple FM cares about is satisfied:
    //
    //   1. macOS 26+ (compile-time @available + runtime check).
    //   2. SystemLanguageModel reports `.available` - device hardware
    //      is eligible AND Apple Intelligence is on AND the on-device
    //      model has finished downloading.
    //   3. SystemLanguageModel.default.supportsLocale() == true - the
    //      current app/system locale is one Apple FM understands.
    @available(macOS 26.0, *)
    static var appleFoundationAvailable: Bool {
        #if canImport(FoundationModels) && !DISABLE_AFM
        let model = SystemLanguageModel.default
        guard case .available = model.availability else { return false }
        return model.supportsLocale()
        #else
        return false
        #endif
    }

    // MARK: - Summarize

    @discardableResult
    func summarize(transcriptURL: URL) async throws -> URL {
        let raw = try String(contentsOf: transcriptURL, encoding: .utf8)

        // Guard: a recording with almost no speech is not a meeting worth
        // minuting. Handed such a transcript the model invents a meeting,
        // so write an honest stub instead of running a backend.
        let wordCount = Self.transcriptWordCount(in: raw)
        if wordCount < Self.minWordsToSummarize {
            log.info(
                "Summary stub: transcript body has \(wordCount, privacy: .public) words (< \(Self.minWordsToSummarize, privacy: .public))."
            )
            let outputURL = Self.sidecarURL(for: transcriptURL)
            try Self.shortTranscriptDocument(transcriptURL: transcriptURL)
                .data(using: .utf8)!.write(to: outputURL, options: .atomic)
            log.info("Summary stub written -> \(outputURL.path, privacy: .public)")
            return outputURL
        }

        let transcript = Self.truncate(transcript: raw)
        let roster = Self.speakerRoster(in: raw)
        if transcript.count != raw.count {
            log.info(
                "Transcript truncated from \(raw.count, privacy: .public) to \(transcript.count, privacy: .public) chars before summarization."
            )
        }

        let backend = Self.currentBackend()
        let summary: String
        let backendLabel: String

        switch backend {
        case .appleFoundation:
            #if canImport(FoundationModels) && !DISABLE_AFM
            if #available(macOS 26.0, *) {
                do {
                    summary = try await runAppleFoundation(
                        transcript: transcript, roster: roster)
                    backendLabel = "Apple Foundation Model"
                    break
                } catch {
                    // Apple FM has two soft failures we recover from rather than
                    // surface raw:
                    //   * locale — it rejects the device language at generation
                    //     time even when availability said .available.
                    //   * content block — the safety guardrail false-positives on
                    //     benign material (e.g. defense news). Under guided
                    //     generation this arrives as .guardrailViolation OR
                    //     .refusal, so we treat both the same.
                    // For either, fall back to Gemma when it is installed;
                    // otherwise surface an honest, specific reason.
                    //
                    // Classify via the TYPED error, never String(describing:):
                    // for a bridged NSError that yields a generic "operation
                    // couldn't be completed (…GenerationError error N.)" form
                    // with neither the case name nor the reason, so the previous
                    // string match silently missed every guardrail trip. The
                    // discriminating signal lives in the typed case (and, as a
                    // defensive fallback, localizedDescription).
                    let isLocale: Bool
                    let isContentBlock: Bool
                    if let generationError = error as? LanguageModelSession.GenerationError {
                        switch generationError {
                        case .unsupportedLanguageOrLocale:
                            isLocale = true
                            isContentBlock = false
                        case .guardrailViolation, .refusal:
                            isLocale = false
                            isContentBlock = true
                        default:
                            isLocale = false
                            isContentBlock = false
                        }
                    } else {
                        let message = error.localizedDescription.lowercased()
                        isLocale =
                            message.contains("unsupported language")
                            || message.contains("unsupported locale")
                            || message.contains("language or locale")
                        isContentBlock =
                            !isLocale && (message.contains("guardrail") || message.contains("unsafe"))
                    }
                    if isLocale || isContentBlock, LLMModelManager.isInstalled() {
                        let kind = isLocale ? "locale" : "guardrail"
                        log.info(
                            "Apple FM \(kind, privacy: .public) rejection; retrying with Gemma fallback."
                        )
                        summary = try await runLlamaCpp(transcript: transcript)
                        backendLabel = "Gemma 3 4B Instruct (Q4_K_M, \(kind) fallback)"
                        break
                    }
                    if isLocale { throw SummarizerError.unsupportedLocale }
                    if isContentBlock {
                        throw SummarizerError.contentFlagged(error.localizedDescription)
                    }
                    throw SummarizerError.generationFailed(error)
                }
            }
            #endif
            // Fell through compile-out or available-check.
            throw SummarizerError.backendUnavailable

        case .llamaCpp(let installed):
            // Defence in depth: currentBackend() already filters out the
            // uninstalled case, but if a future change ever leaks one
            // through, refuse rather than triggering a 2.5 GB download
            // from a meeting-stop code path.
            guard installed else { throw SummarizerError.backendUnavailable }
            summary = try await runLlamaCpp(transcript: transcript)
            backendLabel = "Gemma 3 4B Instruct (Q4_K_M, llama.cpp)"

        case .none:
            throw SummarizerError.backendUnavailable
        }

        let trimmed = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw SummarizerError.emptyResponse }

        let document = sidecarDocument(
            summary: trimmed,
            transcriptURL: transcriptURL,
            backendLabel: backendLabel
        )
        let outputURL = Self.sidecarURL(for: transcriptURL)
        try document.data(using: .utf8)!.write(to: outputURL, options: .atomic)
        log.info("Summary written -> \(outputURL.path, privacy: .public)")
        return outputURL
    }

    // MARK: - Backend implementations

    @available(macOS 26.0, *)
    private func runAppleFoundation(transcript: String, roster: [String]) async throws -> String {
        #if canImport(FoundationModels) && !DISABLE_AFM
        let session = LanguageModelSession(instructions: { Instructions(appleInstructions) })
        let prompt = appleUserPrompt(transcript: transcript, roster: roster)
        // Guided generation: the model fills a `MeetingMinutes` value via
        // constrained decoding rather than emitting free-form Markdown. The
        // output shape is the Swift type's schema, not a template in the
        // prompt, so the model structurally cannot echo a placeholder block
        // back as content - the failure mode of every prompt-only attempt.
        // Greedy sampling keeps the summary deterministic and faithful.
        let response = try await session.respond(
            to: prompt,
            generating: MeetingMinutes.self,
            options: GenerationOptions(sampling: .greedy)
        )
        return Self.renderMarkdown(response.content)
        #else
        throw SummarizerError.backendUnavailable
        #endif
    }

    // System instructions for the Apple FM path. The output *shape* is the
    // `MeetingMinutes` schema (guided generation), so these instructions only
    // govern behaviour - what to extract, what never to invent - not format.
    private var appleInstructions: String {
        """
        You generate meeting minutes from a transcript that has speaker labels.

        Use ONLY facts explicitly stated in the transcript. Never invent \
        decisions, action items, attendees, topics, numbers, or dates. When \
        the transcript contains nothing for a field, leave it empty rather \
        than guessing. Attribute each action item only to a speaker label \
        that appears in the transcript.
        """
    }

    // The Apple FM user turn: the speaker-roster constraint, a one-line
    // directive, and the transcript. The output shape is the MeetingMinutes
    // schema (guided generation), so the user turn carries no format template.
    private func appleUserPrompt(transcript: String, roster: [String]) -> String {
        let constraint = Self.rosterConstraint(roster)
        let header = constraint.isEmpty ? "" : "\(constraint)\n\n"
        return """
            \(header)Summarize the following meeting transcript.

            Transcript:
            \"\"\"
            \(transcript)
            \"\"\"
            """
    }

    #if canImport(FoundationModels) && !DISABLE_AFM
    // Renders the structured minutes into the Markdown sidecar body. Empty
    // collections become `_None._` here, in Swift - the model never sees a
    // template, so it cannot fabricate placeholder content to fill one.
    @available(macOS 26.0, *)
    private static func renderMarkdown(_ minutes: MeetingMinutes) -> String {
        let tldr = minutes.tldr.trimmingCharacters(in: .whitespacesAndNewlines)
        return """
            ## TL;DR

            \(tldr.isEmpty ? "_None._" : tldr)

            ## Key Decisions

            \(bulletList(minutes.keyDecisions))

            ## Action Items

            \(actionItemList(minutes.actionItems))

            ## Topics Discussed

            \(bulletList(minutes.topicsDiscussed))

            ## Open Questions

            \(bulletList(minutes.openQuestions))
            """
    }

    private static func bulletList(_ items: [String]) -> String {
        let cleaned =
            items
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !cleaned.isEmpty else { return "_None._" }
        return cleaned.map { "- \($0)" }.joined(separator: "\n")
    }

    @available(macOS 26.0, *)
    private static func actionItemList(_ items: [MeetingActionItem]) -> String {
        let rendered = items.compactMap { item -> String? in
            let task = item.task.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !task.isEmpty else { return nil }
            let owner = item.owner.trimmingCharacters(in: .whitespacesAndNewlines)
            let due = item.due.trimmingCharacters(in: .whitespacesAndNewlines)
            var meta: [String] = []
            if !owner.isEmpty { meta.append("Owner: \(owner)") }
            if !due.isEmpty { meta.append("Due: \(due)") }
            return meta.isEmpty
                ? "- [ ] \(task)"
                : "- [ ] \(task) (\(meta.joined(separator: ", ")))"
        }
        guard !rendered.isEmpty else { return "_None._" }
        return rendered.joined(separator: "\n")
    }
    #endif

    // Distinct speaker labels that actually appear in the transcript, in order
    // of first appearance. Drives the "do not invent speakers" constraint.
    static func speakerRoster(in transcript: String) -> [String] {
        var roster: [String] = []
        transcript.enumerateLines { line, _ in
            guard line.hasPrefix("**Speaker "), let close = line.range(of: ":**") else {
                return
            }
            let label = String(
                line[line.index(line.startIndex, offsetBy: 2)..<close.lowerBound])
            if !roster.contains(label) {
                roster.append(label)
            }
        }
        return roster
    }

    // One sentence pinning the model to the real speaker set. Empty when the
    // transcript carries no speaker labels (the no-diarization fallback).
    private static func rosterConstraint(_ roster: [String]) -> String {
        guard !roster.isEmpty else { return "" }
        let noun = roster.count == 1 ? "speaker" : "speakers"
        return
            "This transcript has exactly \(roster.count) \(noun): "
            + "\(roster.joined(separator: ", ")). Attribute statements only to these "
            + "labels and never mention any other speaker."
    }

    // Word count of the transcript body - everything after the Markdown
    // header's `---` rule. Drives the too-short-to-summarize guard, and
    // works whether or not the transcript carries `**Speaker N:**` labels.
    static func transcriptWordCount(in transcript: String) -> Int {
        let body: Substring
        if let separator = transcript.range(of: "\n---\n") {
            body = transcript[separator.upperBound...]
        } else {
            body = transcript[...]
        }
        return body.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" }).count
    }

    private func runLlamaCpp(transcript: String) async throws -> String {
        let prompt = chatPrompt(transcript: transcript)

        // Run the generation against a timeout. LlamaRuntime serialises the
        // shared session and runs the decode off the main thread; cancelling
        // `work` makes LlamaSession.generate throw .cancelled at the next
        // sampling step. A corrupt/missing GGUF surfaces as modelLoadFailed,
        // kept distinct from a generation failure.
        let work = Task<String, Error> {
            do {
                return try await LlamaRuntime.shared.generate(
                    prompt: prompt, parameters: .init())
            } catch let error as LlamaSession.LlamaError {
                if case .modelLoadFailed = error {
                    throw SummarizerError.modelLoadFailed(error)
                }
                throw error
            }
        }
        let watchdog = Task<Void, Error> {
            try await Task.sleep(for: Self.generationTimeout)
            work.cancel()
        }

        do {
            let response = try await work.value
            watchdog.cancel()
            return response
        } catch {
            watchdog.cancel()
            if error is CancellationError || (error as? LlamaSession.LlamaError)?.isCancelled == true {
                log.error("Llama generation timed out after \(Self.generationTimeout) - aborting.")
                throw SummarizerError.generationFailed(
                    NSError(
                        domain: "MeetingSummarizer",
                        code: -1,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Generation took longer than \(Int(Self.generationTimeout.components.seconds)) s and was aborted."
                        ]
                    ))
            }
            if error is SummarizerError { throw error }
            throw SummarizerError.generationFailed(error)
        }
    }

    // MARK: - Output paths

    static func sidecarURL(for transcriptURL: URL) -> URL {
        let base = transcriptURL.deletingPathExtension().path
        return URL(fileURLWithPath: base + ".summary.md")
    }

    // MARK: - Transcript truncation

    // Shrink very long transcripts to the maxTranscriptChars budget
    // while keeping the most informative regions: the opening (attendees
    // and agenda) and the closing (decisions and action items). The
    // middle is replaced with an explicit elision marker so the model
    // knows what was dropped.
    private static func truncate(transcript: String) -> String {
        if transcript.count <= maxTranscriptChars { return transcript }
        let head = maxTranscriptChars * 6 / 10
        let tail = maxTranscriptChars - head - 60  // leave room for the marker
        let start = transcript.prefix(head)
        let end = transcript.suffix(max(tail, 200))
        return "\(start)\n\n_…middle of transcript truncated for length…_\n\n\(end)"
    }

    // ------------------------------------------------------------------
    // Gemma prompts (llama.cpp path), wrapped in Gemma 3's chat template.
    // The Apple FM path uses guided generation - see appleInstructions
    // and appleUserPrompt above.
    // ------------------------------------------------------------------

    // System prompt baked with anti-hallucination guardrails. Small
    // models happily invent attendees and action items if you don't pin
    // them to the source text.
    private var systemPrompt: String {
        """
        You are a meeting minutes generator. Given a transcript with speaker
        labels, produce structured Minutes of Meeting in Markdown.

        Rules:
        - Use ONLY facts present in the transcript. Do not invent attendees,
          decisions, or action items.
        - If a section has no content, write "_None._".
        - Attribute action items to the speaker who committed to them, by
          their speaker label (e.g. "Speaker 1").
        - Be concise. Bullet points over paragraphs.
        - Output Markdown only. No preamble, no closing remarks.
        """
    }

    private func userPrompt(transcript: String) -> String {
        """
        Transcript:
        \(transcript)

        Produce these sections in this exact order, in Markdown:

        ## TL;DR
        Two or three sentences.

        ## Key Decisions
        Bulleted list. Each bullet states the decision and any
        deadlines, owners, or budgets.

        ## Action Items
        Checkbox bullets in the form: - [ ] task (Owner: Speaker N, Due: <when if stated>)

        ## Topics Discussed
        Bulleted list of topics with one-line recaps.

        ## Open Questions
        Bulleted list of unresolved questions or follow-ups.
        """
    }

    // Gemma 3's chat template. Gemma doesn't have a separate `system`
    // role - the system prompt rides inline at the top of the user turn.
    // The BOS token is added automatically by llama_tokenize(add_bos:
    // true), so we don't repeat it here. <end_of_turn> closes the user
    // turn; the bare <start_of_turn>model trailer cues generation.
    private func chatPrompt(transcript: String) -> String {
        """
        <start_of_turn>user
        \(systemPrompt)

        \(userPrompt(transcript: transcript))<end_of_turn>
        <start_of_turn>model

        """
    }

    private func sidecarDocument(summary: String, transcriptURL: URL, backendLabel: String) -> String {
        let date = Self.formatTimestamp(Date())
        return """
            # Meeting Summary - \(date)
            _Generated locally by \(backendLabel)._
            _Source transcript: \(transcriptURL.lastPathComponent)_

            ---

            \(summary)
            """
    }

    // Sidecar written when a recording is too short to summarize. An honest
    // stub rather than a fabricated set of minutes - see `minWordsToSummarize`.
    private static func shortTranscriptDocument(transcriptURL: URL) -> String {
        """
        # Meeting Summary - \(formatTimestamp(Date()))
        _Source transcript: \(transcriptURL.lastPathComponent)_

        ---

        _This recording is too short to generate minutes of meeting._
        """
    }

    private static func formatTimestamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm"
        return f.string(from: date)
    }

}

extension LlamaSession.LlamaError {
    fileprivate var isCancelled: Bool {
        if case .cancelled = self { return true }
        return false
    }
}

#if canImport(FoundationModels) && !DISABLE_AFM

// Structured meeting minutes the Apple Foundation Model fills via guided
// generation. The model emits an instance of this type directly through
// constrained decoding - it never sees a Markdown template, so it cannot
// echo placeholder text back as content. Empty arrays render as `_None._`
// deterministically in `MeetingSummarizer.renderMarkdown`.
@available(macOS 26.0, *)
@Generable
struct MeetingMinutes {
    @Guide(description: "A two or three sentence plain overview of what actually happened. Do not embellish.")
    let tldr: String

    @Guide(description: "Decisions explicitly made during the meeting. Empty when the transcript records none.")
    let keyDecisions: [String]

    @Guide(description: "Action items someone explicitly committed to. Empty when there are none.")
    let actionItems: [MeetingActionItem]

    @Guide(description: "Distinct topics actually discussed. Empty when none.")
    let topicsDiscussed: [String]

    @Guide(description: "Questions raised but left unresolved. Empty when none.")
    let openQuestions: [String]
}

@available(macOS 26.0, *)
@Generable
struct MeetingActionItem {
    @Guide(description: "What needs to be done.")
    let task: String

    @Guide(description: "Speaker label of the owner, exactly as written in the transcript.")
    let owner: String

    @Guide(description: "When it is due. Empty string when the transcript states no due date.")
    let due: String
}

#endif
