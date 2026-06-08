# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoneyPrinterV2 (MPV2) is a Python 3.12 CLI tool that automates four online workflows:
1. **YouTube Shorts** — generate video (LLM script → TTS → images → MoviePy composite) and upload via Selenium
2. **Twitter/X Bot** — generate and post tweets via Selenium
3. **Affiliate Marketing** — scrape Amazon product info, generate pitch, share on Twitter
4. **Local Business Outreach** — scrape Google Maps (Go binary), extract emails, send cold outreach via SMTP

There is no web UI, no REST API, no test suite, no CI, and no linting config.

## Running the Application

```bash
# First-time setup
cp config.example.json config.json   # then fill in values
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# macOS quick setup (auto-configures Ollama, ImageMagick, Firefox profile)
bash scripts/setup_local.sh

# Preflight check (validates services are reachable)
python scripts/preflight_local.py

# Run
python src/main.py
```

The app **must** be run from the project root. `python src/main.py` adds `src/` to `sys.path`, so all imports use bare module names (e.g., `from config import *`, not `from src.config import *`).

## Architecture

### Entry Points
- `src/main.py` — interactive menu loop (primary)
- `src/cron.py` — headless runner invoked by the scheduler as a subprocess: `python src/cron.py <platform> <account_uuid>`

### Provider Pattern
Two service categories use a string-based dispatch pattern configured in `config.json`:

| Category | Config key | Options |
|---|---|---|
| LLM | `ollama_model` | Ollama (via `ollama` Python SDK). If empty, user picks from available models at startup. |
| Image gen | — | `nanobanana2` (Gemini image API) |
| STT | `stt_provider` | `local_whisper`, `third_party_assemblyai` |

LLM always uses the local Ollama server. Image generation always uses Nano Banana 2.

### Key Modules
- **`src/llm_provider.py`** — unified `generate_text(prompt)` function using the Ollama Python SDK
- **`src/config.py`** — 30+ getter functions, each re-reads `config.json` on every call (no caching). `ROOT_DIR` = project root, computed as `os.path.dirname(sys.path[0])`
- **`src/cache.py`** — JSON file persistence in `.mp/` directory (accounts, videos, posts, products)
- **`src/constants.py`** — menu strings, Selenium selectors (YouTube Studio, X.com, Amazon)
- **`src/classes/YouTube.py`** — most complex class; full pipeline: topic → script → metadata → image prompts → images → TTS → subtitles → MoviePy combine → Selenium upload
- **`src/classes/Twitter.py`** — Selenium automation against x.com
- **`src/classes/AFM.py`** — Amazon scraping + LLM pitch generation
- **`src/classes/Outreach.py`** — Google Maps scraper (requires Go) + email sending via yagmail
- **`src/classes/Tts.py`** — KittenTTS wrapper

### Data Storage
All persistent state lives in `.mp/` at the project root as JSON files (`youtube.json`, `twitter.json`, `afm.json`). This directory also serves as scratch space for temporary WAV, PNG, SRT, and MP4 files — non-JSON files are cleaned on each run by `rem_temp_files()`.

### Browser Automation
Selenium uses pre-authenticated Firefox profiles (never handles login). The profile path is stored per-account in the cache JSON and also in `config.json` as a default.

### CRON Scheduling
Uses Python's `schedule` library (in-process, not OS cron). The scheduled job spawns `subprocess.run(["python", "src/cron.py", platform, account_id])`.

## Known environment issue: Pillow/freetype segfault
The Pillow build in this project's venv (9.5.0, pinned for MoviePy 1.0.3's
`Image.ANTIALIAS`) segfaults nondeterministically on repeated freetype text
rendering — `textbbox`/`draw.text` with multi-word strings, multiple font faces,
or `getlength` after `draw` all trip heap corruption that accumulates across
calls. The long-form thumbnail overlay was moved off Pillow onto ImageMagick
because of this. **The Shorts subtitle path (`src/classes/YouTube.py`, which uses
MoviePy `TextClip` → ImageMagick) likely shares the same underlying fragility if
it ever falls back to Pillow text;** if Shorts subtitles start crashing, suspect
this and prefer ImageMagick-based rendering there too.

## Configuration

All config lives in `config.json` at the project root. See `config.example.json` for the full template and `docs/Configuration.md` for reference. Key external dependencies to configure:
- **ImageMagick** — required for MoviePy subtitle rendering (`imagemagick_path`)
- **Firefox profile** — must be pre-logged-in to target platforms (`firefox_profile`)
- **Ollama** — for LLM text generation (via `ollama` Python SDK)
- **Nano Banana 2** — for image generation (Gemini image API)
- **Go** — only needed for Outreach (Google Maps scraper)

## Contributing

PRs go against `main`. One feature/fix per PR. Open an issue first. Use `WIP` label for in-progress PRs.

## Long-form engine (standalone, no brand) — `src/longform/`

A separate faceless long-form YouTube engine lives under `src/longform/`. It is
self-contained (depends only on `requests` + stdlib) and shares no logic with
the Shorts pipeline.

Rules:
- The scout/farmer chooses niche AND format from live YouTube data. Never
  hardcode a topic; niche probes in `longform_discovery.json` are seeds, not a
  fixed list.
- HARD LEGAL GATE (future render stages): no clip enters a render unless its
  source is in `longform_discovery.json` `legal_sources` AND its
  `{source, source_id, license, attribution_required}` is logged. The data
  farmer records `cc_feasibility` per niche as an early read on sourceability.
- READ vs WRITE keys are separate: the farmer uses a Data API **key**
  (`longform.youtube_api_key` / `YOUTUBE_API_KEY`); publishing uses the OAuth
  upload client. Never mix them.
- Keep farming non-blocking: API failures retry with backoff; the quota guard
  stops gracefully before the daily budget and persists partial results.
- Never publish directly; everything ultimately goes through
  `review_queue/approved/`.

### Data farmer (this layer)
- `src/longform/youtube_api.py` — quota-aware Data API v3 client (search.list
  = 100 units, videos/channels.list = 1; default budget 9000/day).
- `src/longform/farmer.py` — farms each niche probe; keeps videos > 30 min and
  > 1M views; computes `view_velocity` (views/day) and `outlier_score`
  (views/subs); scores with `scout_scoring.weights`; applies
  `min_score_to_queue`.
- `src/longform/storage.py` — SQLite `.mp/farm.db`, idempotent upserts.
- `src/longform/title_match.py` — matches winners against `title_formulas.json`.
- Outputs: `topics.longform.json` (ranked slate) + `scripts/farm_report.py`.
- Run a live farm: `python scripts/run_farm.py` (from repo root) or
  `cd src && python -m longform.farmer`. The package uses relative imports, so
  do not run `src/longform/farmer.py` as a loose script.

### Image providers — `src/longform/image_providers.py`
The engine must never hard-depend on one paid image backend. `generate_image(
prompt, aspect_ratio, output_path, quality="standard")` tries a chosen provider
and falls back down a configurable chain. Backends:
- `pollinations` (DEFAULT) — free, keyless. Zero setup. Anonymous access is
  rate-limited (~1 req/15s) and shared/datacenter IPs may be hard-blocked with
  HTTP 402; set `image.pollinations.referrer` or `token` (free at
  enter.pollinations.ai) to lift the limit for headless/automation use.
- `cloudflare` — Workers AI Flux (free tier); needs `image.cloudflare.account_id`
  + `api_token`.
- `local_sd` — optional local Stable Diffusion HTTP API at `image.local_sd_url`.
- `gemini` — paid; used ONLY when `quality="high"` AND a Gemini key is set. It
  reuses the existing `nanobanana2_api_key` (one Gemini key, never duplicated).

Rules:
- Fallback never crashes a render: if every eligible provider fails, it writes a
  flagged colored placeholder PNG (`is_placeholder=True`).
- Free backends cost 0; every call is logged to the `image_log` table in
  `.mp/farm.db` (provider + cost) via `storage.log_image`.
- Self-contained: `requests` + stdlib only (placeholder PNG is hand-encoded; no
  Pillow). Config is resolved by `config.get_image_config()`.
- Packaging/render stages (future `packaging.py`) MUST call `generate_image()`
  rather than any backend directly, so swapping providers needs no code change.
  `image.thumbnail_provider` lets thumbnails use a higher-quality backend
  without touching bulk stills (pass it as the `provider=` arg).

### LLM provider — `src/longform/llm.py`
Pluggable behind one `llm(prompt)->str` interface (`config.llm`):
- `provider: openai_compatible` — any OpenAI-style `/chat/completions` endpoint.
  **Groq's FREE tier is the recommended default** for script quality
  (`base_url https://api.groq.com/openai/v1`, `model llama-3.3-70b-versatile` —
  ~20x the local 3B). The key is read from the env var named by
  `llm.openai_compatible.api_key_env` (default `GROQ_API_KEY`), never stored in
  config.
- `provider: ollama` — local Ollama HTTP API; the offline fallback.
- Non-blocking fallback: if the primary provider errors (no key, network), the
  call falls back to local Ollama rather than crashing a run.
- Local helpers detect installed Ollama models (`/api/tags`) + system RAM and
  recommend the largest viable model (floor `qwen2.5:7b-instruct` /
  `llama3.1:8b`). `scripts/llm_doctor.py` prints this; weights are NEVER pulled
  automatically — confirm before `ollama pull`.

### Creative layer (iceberg track) — script / hooks / packaging
The creative layer turns a chosen topic into a review-ready package. The LLM is
pluggable (see above; defaults to Groq, falls back to local Ollama) and every
generator takes an injectable `llm(prompt)->str` so tests never hit a server.
Few-shot tone comes from the REAL top `iceberg_deepdive` winners in
`.mp/farm.db`, never generic priors.
- `prompts/longform_iceberg.yaml` — versioned script-architect prompt (bump
  `version` on edit; recorded in the ledger per script). Current: `iceberg-v2`.
- `src/longform/script.py` — **staged** generation: (a) OUTLINE pass (tier
  skeleton + titles + premises + cold_hook + final_payoff), (b) PER-ENTRY pass
  generating each tier to its own word budget (`target_minutes*140 /
  entry_count`) with a running-context summary for coherence, (c) LENGTH
  enforcement (expand if under ~85% of budget, cap 3 retries), (d) reassemble
  into the cold_hook→tiers→final_payoff schema. Logs prompt_version,
  entry_count, word_count, and sources.
- `src/longform/research.py` — optional factual GROUNDING (`config.script.
  grounding`, default on): pulls REAL candidate entries + facts (free Wikipedia
  REST API, keyless) into the outline + per-entry prompts so narration is
  grounded, not invented. Non-blocking; falls back to model-only. Sources used
  are stored on the `creative_runs` row.
- `src/longform/hooks.py` — 6-10 cold-open variants, LLM self-scored on
  curiosity / depth-pull / payoff-promise; best kept; all variants + scores
  logged.
- `src/longform/packaging.py` — 5 titles (iceberg + specific_number +
  curiosity_gap) and 3 tiered thumbnail concepts; thumbnails are rendered by
  calling `image_providers.generate_image()` (free Pollinations path) at
  1280x720 then overlaying big title text with ImageMagick (the local Pillow
  build segfaults on repeated freetype text rendering; ImageMagick rasterizes in
  its own process and is the project's existing text renderer). Writes ONE
  review item to
  `review_queue/pending/<run_id>/` (gitignored) and logs the chosen title/hook.
- Creative ledger tables live in `.mp/farm.db`: `creative_runs`, `hook_variants`,
  `title_variants`.
- This layer may use PyYAML + Pillow (render-side); the data farmer core
  (`youtube_api`, `farmer`, `storage`, `image_providers`) stays requests+stdlib.

### Reference-exemplar layer (learned style) — exemplars / style_spec
Instead of imitating generic "be dramatic" priors, the creative layer learns the
genre's winning pattern from REAL top performers farmed into `.mp/farm.db`.
- `src/longform/exemplars.py` — selects top videos for a niche by a blended rank
  (views + view_velocity + outlier_score), runs the existing LLM relevance
  filter FIRST so off-topic bleed is excluded, and enforces ANTI-OVERFIT by
  sampling across many channels (per-channel cap, default 2). For each kept video
  it fetches the transcript (free `youtube-transcript-api`), derives the opening
  hook, rough structure, chapter markers (from the description via the Data API
  when available), and pacing. Transcripts are cleaned of ASR noise. Fully
  non-blocking: a video with no transcript is skipped and the next is pulled.
- `src/longform/style_spec.py` — an LLM distills the cleaned exemplars into a
  STRUCTURED per-niche style spec (hook formula, narrator tone, pacing, dread
  build, tier transitions, words-per-entry, escalation, title patterns, chapter
  style). Cached at `.mp/style_spec_<niche>.json`; rebuild with `refresh=True`.
  A clearly-marked `rerank_by_retention()` hook lets the spec become
  self-improving once the retention loop exists.
- **LEGAL GUARDS (enforced in code, not just prose):** the spec stores ABSTRACT
  patterns plus at most very short (<=15 word) illustrative snippets — NEVER full
  transcripts (harvested transcripts are transient, never written to review
  outputs). Generation must never reproduce exemplar wording or borrow their
  facts; our facts come ONLY from our own Wikipedia grounding (`research.py`).
  `style_spec.find_leak()` detects any long verbatim run (>=8 words) shared with
  an exemplar; `sanitize_spec()` scrubs the spec before caching, and a test
  asserts no exemplar substring can leak into a generated script.
- INJECTION: the cached spec feeds Pass 2 of the two-pass script (learned hook
  formula / pacing / transitions / escalation), `hooks.py` (hook formula),
  `packaging.build_titles_with_spec` (title patterns), and is surfaced for
  chapter naming. Only the CACHED spec is read during script generation (no live
  harvest mid-run); build it separately. Prompt bumped to `iceberg-v4`.

### Production layer (script -> 1080p MP4) — tts / sourcer / compose
Turns a finished script into a real 16:9 1080p MP4. **Requires ffmpeg** (set
`ffmpeg_path` in config, or have it on PATH); `scripts/produce_video.py
<run_id>` renders a review item. Everything is non-blocking — a single
API/clip/caption failure falls back and is logged; a render never crashes.
- `src/longform/tts.py` — provider interface; default `edge_tts` (free, keyless;
  exposes WordBoundary timings for caption sync), optional `elevenlabs`. Falls
  back to a silent track + estimated word timings. Config: `tts.provider`,
  `tts.voice`.
- `src/longform/sourcer.py` — per shot, queries `legal_sources` (Pexels/Pixabay
  with keys; Openverse keyless), filters to commercial-safe licenses, downloads,
  and perceptual-hash dedupes so no clip repeats. Falls back to an AI image
  (`image_providers.generate_image`) then a colored slate. HARD RULE: every
  asset is logged to the `asset_log` ledger with {source, source_id, url,
  license, attribution_required}; `storage.license_manifest()` flags any
  unlicensed asset. Stock keys come from env (`PEXELS_API_KEY`/`PIXABAY_API_KEY`).
- `src/longform/compose.py` — 1080p compositor: per-shot clips timed to the
  section VO (Ken-Burns on stills), cross-dissolves, captions burned from TTS
  word timings via **ImageMagick** (never Pillow), ducked royalty-free music from
  `assets/music/`, YouTube chapter markers embedded + a description chapter list,
  and an auto-appended credits slate for attribution-required assets. Output MP4
  + `production_metadata.json` go to `review_queue/pending/<run_id>/`. Nothing
  publishes.
