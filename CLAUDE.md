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

## Configuration

All config lives in `config.json` at the project root. See `config.example.json` for the full template and `docs/Configuration.md` for reference. Key external dependencies to configure:
- **ImageMagick** — required for MoviePy subtitle rendering (`imagemagick_path`)
- **Firefox profile** — must be pre-logged-in to target platforms (`firefox_profile`)
- **Ollama** — for LLM text generation (via `ollama` Python SDK)
- **Nano Banana 2** — for image generation (Gemini image API)
- **Go** — only needed for Outreach (Google Maps scraper)

## Contributing

PRs go against `main`. One feature/fix per PR. Open an issue first. Use `WIP` label for in-progress PRs.

## Fork-specific rules (OffSzn content engine)

- NEVER publish directly from a pipeline run. All output goes to
  ./review_queue/pending/; only ./review_queue/approved/ may be published.
- Selenium upload paths are deprecated. Do not extend them. All publishing
  goes through the YouTube Data API, Post Bridge, or the X API.
- Every generated asset MUST get a ledger row in .mp/ledger.db and a
  utm_campaign of the form mpv2-{lane}-{video_id}. No untracked output.
- Prompts live in prompts/*.yaml with versions — never hardcode prompt
  strings in Python.
- Lanes: "offszn" (product content, always CTA + UTM link) and "faceless".
- Store facts (use these, don't invent): OffSzn buds, $24.99 single,
  $49.99 x2, $59.99 x3, chrome design, USB-C, store offszn.us.
- Keep Ollama for text generation. Never add gpt4free or similar
  reverse-engineered API dependencies.

### Phase 1 architecture (review gate, ledger, API publishing)

The generation → distribution flow is now:

1. **Generate** — `main.py` (option 1) or `cron.py` call
   `YouTube.generate_video()`, which creates a ledger row up front and updates
   it as each stage completes. The finished MP4 is submitted to the review
   queue (pending) via `review_queue.submit()`; nothing uploads here.
2. **Review** — `python src/review.py` lists pending items and lets a human
   approve/reject. Approved items move to `review_queue/approved/`.
3. **Publish** — `python src/publish.py` drains `review_queue/approved/`,
   uploading to YouTube (Data API) and/or TikTok+Instagram (Post Bridge),
   then moves items to `review_queue/published/` and marks the ledger row
   `published` with the platform and post id.

Key modules added in Phase 1:
- `src/ledger.py` — SQLite run ledger (`.mp/ledger.db`).
- `src/review_queue.py` — filesystem review-queue gate + ledger integration.
- `src/review.py` — review CLI.
- `src/youtube_upload.py` — YouTube Data API v3 uploader (OAuth installed-app
  flow, token cached at `.mp/youtube_token.json`). Set `use_selenium_upload:
  true` to fall back to the deprecated Selenium path.
- `src/publish.py` — publisher that drains the approved queue.

To use the YouTube Data API path, download an OAuth client secrets file
(Desktop app) from the Google Cloud console and point
`youtube.client_secrets_file` at it (default `client_secret.json` in the repo
root, which is gitignored).
