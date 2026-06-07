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

from longform import compose, image_providers, music, sourcer, storage, tts


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

    def image_fallback_fn(prompt, out_path):
        return image_providers.generate_image(
            prompt, aspect_ratio="16:9", output_path=out_path, quality="standard",
            session=session, db_path=None,
        )

    def source_shot_fn(shot_text, shot_index, section_title, dest_dir):
        return sourcer.source_shot(
            shot_text, shot_index, section_title, run_id, dest_dir,
            providers=sourcer.DEFAULT_PROVIDERS, footage_cfg=footage_cfg,
            image_fallback_fn=image_fallback_fn, used_hashes=used_hashes,
            session=session, ffmpeg_path=ffmpeg_path, db_path=None,
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
    print(f"\nCHAPTERS:\n{report['chapters_text']}")
    print(f"\nmetadata sidecar: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
