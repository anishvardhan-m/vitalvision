# VitalVision - project rules for Claude

This is a learning project. Do NOT write pipeline code (capture, ROI, signal
processing, BPM, evaluation) unless the user explicitly says "write it for me".
Instead: explain the concept, describe what the code should do, then review the
code the user writes.

Keep it simple: plain functions and scripts until something works; refactor
into the `src/vitalvision` package only in Phase 7. No Flask, web servers, or
extra features beyond what the current phase needs.

Scope is heart rate only. Phases:
1. Face detection + ROI selection
2. Raw signal extraction
3. BPM estimation (offline)
4. Real-time pipeline
5. Evaluation (Green/CHROM/POS vs. pulse oximeter ground truth)
6. Demo app
7. Engineering finish (package structure, tests, docs)

Never commit `data/raw/` or model files (`*.task`). Use small commits with
conventional messages (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).

Never write `docs/devlog.md` entries for the user - that file is theirs.

Use `time.monotonic()` for frame timestamps. Use the MediaPipe CPU delegate
(not GPU/Metal - 0.10.35 is pinned because newer MediaPipe crashes with a
Metal error on this M4 Mac).

When committing, never add Co-Authored-By or any Claude/AI attribution to
commit messages. All commits are authored by the user.
