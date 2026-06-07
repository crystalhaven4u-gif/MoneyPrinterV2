#!/usr/bin/env python3
"""
Generate ONE complete iceberg deep-dive package into the review queue.

    python scripts/generate_iceberg_sample.py "Lost Media"

Runs the creative layer end to end with the local Ollama LLM and a real
thumbnail render (free Pollinations path, with graceful fallback), writing a
review item to review_queue/pending/<run_id>/. Nothing publishes.
"""

import os
import sys
from uuid import uuid4

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

for _stream in (sys.stdout, sys.stderr):  # readable on the Windows cp1252 console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from longform import llm, packaging

TARGET_MINUTES = 10
WPM = 140


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    topic = argv[0] if argv else "Lost Media"
    run_id = f"iceberg-{uuid4().hex[:8]}"

    print(f"Provider: {llm.active_provider_label()}")
    print(f"Generating iceberg package for: {topic}  (run_id={run_id})")
    summary = packaging.generate_iceberg_package(
        topic, run_id=run_id, target_minutes=TARGET_MINUTES, n_hooks=6
    )

    script = summary["script"]
    target = TARGET_MINUTES * WPM
    wc = script["word_count"]
    print("\n" + "=" * 72)
    print(f"REVIEW ITEM: {summary['item_dir']}")
    print("=" * 72)
    print(f"prompt_version : {script['prompt_version']}")
    print(f"tiers (entries): {script['entry_count']}")
    print(f"WORD COUNT     : {wc} / {target} target  ({100*wc/target:.0f}%)")
    print(f"grounded       : {script.get('grounded')}  ({len(script.get('sources', []))} sources)")

    print("\nPER-TIER WORD COUNTS:")
    for row in script.get("per_entry_word_counts", []):
        print(f"  {row['word_count']:4d} words ({row['expansions']} expansions)  {row['entry_title']}")

    print("\nSOURCES KEPT (grounding):")
    for url in script.get("sources", []):
        print(f"  + {url}")
    if not script.get("sources"):
        print("  (none — grounding returned nothing or was throttled; model-only fallback)")

    rejected = script.get("rejected_sources", [])
    print(f"\nSOURCES REJECTED by relevance filter ({len(rejected)}):")
    for item in rejected:
        print(f"  - {item.get('title')} ({item.get('url')}) :: {item.get('reason')}")
    if not rejected:
        print("  (none rejected)")

    print(f"\nCHOSEN TITLE   : {summary['chosen_title']}")
    print(f"CHOSEN HOOK    : {summary['chosen_hook']}")
    print("\nALL 5 TITLE VARIANTS:")
    for variant in packaging.build_title_variants(topic, entry_count=script["entry_count"]):
        print(f"  [{variant['formula']:16}] {variant['title']}")

    print("\nTHUMBNAILS:")
    for thumb in summary["thumbnails"]:
        print(f"  - {thumb['concept_id']:16} provider={thumb['provider']} "
              f"placeholder={thumb['is_placeholder']}  {thumb['path']}")

    print("\n" + "=" * 72)
    print("FULL SCRIPT")
    print("=" * 72)
    print(f"\n[COLD HOOK]\n{script['cold_hook']}\n")
    for tier in script["tiers"]:
        print(f"[T{tier['tier']} — {tier['label']}: {tier['entry_title']}]")
        print(tier["narration"])
        print(f"  (shots: {', '.join(tier['shot_list'])})")
        print(f"  (open-loop: {tier['open_loop']})\n")
    print(f"[FINAL PAYOFF]\n{script['final_payoff']}")
    print(f"\nFull package JSON: {summary['package_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
