# Real-model recording — 29 September 2026

[Watch the full recording](copilot-demo.mp4) · [GIF highlights](copilot-demo.gif) · [Poster](copilot-demo-poster.png)

Gemma 3 4B (`gemma3:4b`, Q4_K_M, model ID `a2af6cc3eb7f`) through Ollama 0.34.4 on an Apple M5 / 32 GiB Mac. The recorded research answer has `usage.provider=ollama`, `usage.model=gemma3:4b`, one supporting citation, and no validation-fallback limitation. Backend-reported latency is 4,976 ms. See the [actual persisted response](demo-session.json), [SEC corpus manifest](demo-corpus.json), and [media hashes/edit decisions](media-metadata.json).

## Included shots

- 0–8 s: Apple, 12 ready SEC filings and 226 chunks; visible 10-K and latest 10-Q names, dates and chunk counts.
- 8–29 s: submit a Services sales / gross-margin question and show the cited model answer. The five-second generation wait is sped up 4x and labelled. The application returns the answer at once; it does not stream.
- 29–43 s: open the citation, show its filing/section metadata, then scroll to the exact Services growth sentence in Apple's 2025 Form 10-K.
- 43–52 s: show `gemma3:4b`, `ollama`, latency and the answer's limitations. This is the accepted model response, not deterministic synthesis.
- 52–78 s: ask whether to buy AAPL / request a price target, then show the explicit refusal. **This is an application guardrail before inference, not a model-generated refusal or a validation fallback.**

The MP4 is 78.27 seconds, 1920×1406 (native window aspect scaled to 1920 wide), 30 fps, H.264, no audio, 2.16 MB. The GIF is a 14-second chronological highlight edit at 960×703, 1.62 MB. The full-size poster is 0.65 MB. File sizes use decimal MB.

## Scope and caveats

The corpus was built through the app's SEC EDGAR flow with a configured fair-access User-Agent. Apple was used instead of the sample MSFT/NVDA companies. Its 12 imported filings include the annual 10-K and recent 10-Qs; the full inventory and source hashes are in the manifest. The focused recorded question uses the visible chat retrieval control set to one passage; the evaluation uses the default eight. No source text, answer, or API metadata was manually substituted for the recording.

The three Services driver/margin statements are supported by the opened chunk. Accepted citation syntax is not a proof of semantic correctness. The model's visible limitation about absent specific figures is too broad: the full source chunk contains a gross-margin table. That imperfection is retained. The HTML importer represents flattened text as page 1; the excerpt itself retains the original printed Form 10-K page marker (23). Fiscal-year labels in the imported document list are importer metadata, not independently normalized fiscal-period labels.

Earlier rehearsals exposed retrieval and citation problems. One initial take was rejected for an overgeneralized product-sales statement; a later capture failed to finalize because of an overlapping recorder. The published file is a fresh continuous take. It is a rehearsed example, not an estimate of typical success. All completed benchmark runs and an interrupted diagnostic are separately retained in [the results history](../../evals/results/2026-09-29/README.md).

Recording used a clean Chrome Guest window and window-only capture, excluding the desktop, Dock, other tabs/windows, and notification overlays. Do Not Disturb activation was not verified. The optional memo/comparison shot was omitted to keep the recording short.

QA: decoded the entire final video without errors; inspected the opening, answer, source, metadata, refusal and end frames plus one-second contact sheets and denser generation-transition frames. No error toast or fallback research answer appears in the published take. The source recording and review frames remain local.
