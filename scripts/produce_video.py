#!/usr/bin/env python3
"""
Produce a 1080p MP4 from a finished iceberg script in the review queue.

    python scripts/produce_video.py iceberg-a4636e49

Wires the production layer: TTS (tts.py) -> footage sourcing (sourcer.py) ->
compositing (compose.py). Reads the script from
review_queue/pending/<run_id>/package.json, sources commercial-safe footage
(logging every license to the ledger), and writes the final MP4 + metadata
(chapters, license manifest, credits) back into that review item. Nothing
publishes.

A per-run PID lockfile (.mp/locks/produce_<run_id>.lock) prevents two concurrent
renders of the same run_id from double-writing the asset ledger; a lock whose
owner process is dead is reclaimed as stale.
"""

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import requests

from longform import compose, image_providers, llm as llm_module, music, sourcer, storage, thumbnail, tts

LOCK_DIR = os.path.join(ROOT_DIR, ".mp", "locks")


def _pid_alive(pid: int) -> bool:
    """Whether a PID is a currently-running process (cross-platform)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _RunLock:
    """A per-run PID lockfile so two concurrent renders can't double-write the
    asset ledger for the same run_id. A lock whose owner PID is dead is treated
    as stale and taken over."""

    def __init__(self, run_id: str):
        self.path = os.path.join(LOCK_DIR, f"produce_{run_id}.lock")
        self.acquired = False

    def acquire(self) -> bool:
        os.makedirs(LOCK_DIR, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode())
                finally:
                    os.close(fd)
                self.acquired = True
                return True
            except FileExistsError:
                try:
                    with open(self.path, "r", encoding="utf-8") as handle:
                        owner = int((handle.read().strip() or "0"))
                except (OSError, ValueError):
                    owner = 0
                if owner and owner != os.getpid() and _pid_alive(owner):
                    return False  # held by a live render
                try:  # stale lock -> remove and retry once
                    os.remove(self.path)
                except OSError:
                    return False
        return False

    def release(self) -> None:
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: produce_video.py <run_id>")
        return 2
    run_id = argv[0]

    item_dir = os.path.join(ROOT_DIR, "review_queue", "pending", run_id)
    package_path = os.path.join(item_dir, "package.json")
    if not os.path.exists(package_path):
        print(f"No package.json at {package_path}")
        return 1
    with open(package_path, "r", encoding="utf-8") as handle:
        package = json.load(handle)
    script = package.get("script") or package

    # Run-lock: refuse to start if another render for this run_id is live, so two
    # processes can't double-write the asset ledger (stale locks are reclaimed).
    lock = _RunLock(run_id)
    if not lock.acquire():
        print(f"Another produce_video is already rendering '{run_id}'.")
        print(f"Lock held: {lock.path}. Aborting to avoid double-writing the ledger.")
        return 3
    try:
        return _render(run_id, item_dir, script)
    finally:
        lock.release()


def _render(run_id, item_dir, script) -> int:
    # Config
    import config as cfg
    ffmpeg_path = cfg.get_ffmpeg_path()
    magick_path = cfg.get_imagemagick_path()
    tts_cfg = cfg.get_tts_config()
    footage_cfg = cfg.get_footage_config()
    music_dir = os.path.join(ROOT_DIR, footage_cfg.get("music_dir", "assets/music"))

    print(f"run_id        : {run_id}")
    print(f"ffmpeg        : {ffmpeg_path}")
    print(f"tts provider  : {tts_cfg.get('provider')} ({tts_cfg.get('voice')})")
    print(f"stock keys    : pexels={'yes' if footage_cfg.get('pexels_api_key') else 'no'} "
          f"pixabay={'yes' if footage_cfg.get('pixabay_api_key') else 'no'}")
    print("Producing video (this can take a while)...\n")

    session = requests.Session()
    used_hashes = set()
    topic = (script.get("iceberg_topic") if isinstance(script, dict) else None) or run_id

    # LLM-backed relevance ranker: fetch many candidates per shot and keep the
    # best visual match (non-blocking -- falls back to provider order on error).
    ranker = sourcer.make_relevance_ranker(llm_module.default_llm())

    def image_fallback_fn(prompt, out_path):
        return image_providers.generate_image(
            prompt, aspect_ratio="16:9", output_path=out_path, quality="standard",
            session=session, db_path=None,
        )

    def _ai_prompt(spec):
        """An eerie, on-theme AI prompt for an atmosphere shot."""
        bits = [spec.get("description") or spec.get("query") or "", topic]
        if spec.get("mood"):
            bits.append(spec["mood"])
        bits += ["dark", "cinematic", "ominous lighting", "photoreal", "16:9"]
        return ", ".join(b for b in bits if b)

    def source_shot_fn(shot, shot_index, section_title, dest_dir):
        spec = shot if isinstance(shot, dict) else {
            "description": str(shot), "query": str(shot), "mood": "", "type": "concrete"
        }
        query = spec.get("query") or spec.get("description") or section_title
        prefer_ai = spec.get("type") == "atmosphere"
        return sourcer.source_shot(
            spec.get("description") or query, shot_index, section_title, run_id, dest_dir,
            providers=sourcer.DEFAULT_PROVIDERS, footage_cfg=footage_cfg,
            image_fallback_fn=image_fallback_fn, used_hashes=used_hashes,
            session=session, ffmpeg_path=ffmpeg_path, db_path=None,
            search_query=query, ai_prompt=_ai_prompt(spec), prefer_ai=prefer_ai,
            ranker=ranker,
        )

    def tts_fn(text, out_path):
        return tts.synthesize_section(text, out_path, cfg=tts_cfg)

    mood = footage_cfg.get("music_mood", "dark ambient")

    def music_fetch_fn():
        return music.fetch_music_bed(mood, music_dir, cfg=footage_cfg, session=session)

    report = compose.produce(
        script, run_id, item_dir, source_shot_fn, tts_fn, music_fetch_fn=music_fetch_fn,
        ffmpeg_path=ffmpeg_path, magick_path=magick_path, music_dir=music_dir,
        db_path=None,
    )

    # Real iceberg thumbnails (3 variants) -- AI base when reachable, else a
    # procedural ImageMagick iceberg so they're never gray placeholders.
    tier_labels = [t.get("label", "") for t in (script.get("tiers") or [])]
    thumbnails = []
    try:
        thumbnails = thumbnail.render_iceberg_thumbnails(
            topic, item_dir, magick_path, tier_labels=tier_labels, session=session,
        )
    except Exception as exc:
        print(f"thumbnail render skipped: {exc}")

    # Metadata sidecar
    manifest = storage.license_manifest(run_id)
    metadata = {
        "run_id": run_id,
        "mp4_path": report["mp4_path"],
        "duration_seconds": report["duration"],
        "chapters_for_description": report["chapters_text"],
        "chapters": report["chapters"],
        "credits": report["credits"],
        "license_manifest_complete": manifest["complete"],
        "unlicensed_assets": manifest["unlicensed"],
        "iceberg_thumbnails": thumbnails,
        "counts": {
            "real_clips": report["real_clips"],
            "ai_fallbacks": report["ai_fallbacks"],
            "slates": report["slates"],
            "failed_shots": report["failed_shots"],
        },
        "assets": manifest["assets"],
    }
    meta_path = os.path.join(item_dir, "production_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n" + "=" * 64)
    print("PRODUCTION REPORT")
    print("=" * 64)
    print(f"MP4            : {report['mp4_path']}")
    print(f"duration       : {report['duration']} s")
    print(f"sections        : {report['sections']}")
    print(f"real clips      : {report['real_clips']}")
    print(f"AI-image fallbk : {report['ai_fallbacks']}")
    print(f"slates          : {report['slates']}")
    print(f"failed shots    : {report['failed_shots']}")
    print(f"license manifest complete: {manifest['complete']} "
          f"({len(manifest['assets'])} assets, {len(manifest['unlicensed'])} unlicensed)")
    track = report.get("music")
    if track:
        print(f"music track     : \"{track.get('title')}\" by {track.get('author') or 'n/a'} "
              f"[{track.get('source')} / {track.get('license')}] {track.get('url')}")
    else:
        print("music track     : (none fetched; local bed or silence)")
    print(f"credits         : {report['credits'] or '(none required)'}")
    if thumbnails:
        print("iceberg thumbs  :")
        for thumb in thumbnails:
            print(f"   - {thumb['variant_id']:18} base={thumb['base_source']:12} {thumb['path']}")
    print(f"\nCHAPTERS:\n{report['chapters_text']}")
    print(f"\nmetadata sidecar: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
