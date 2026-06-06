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

from longform import packaging


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    topic = argv[0] if argv else "Lost Media"
    run_id = f"iceberg-{uuid4().hex[:8]}"

    print(f"Generating iceberg package for: {topic}  (run_id={run_id})")
    summary = packaging.generate_iceberg_package(
        topic, run_id=run_id, target_minutes=10, n_hooks=6
    )

    script = summary["script"]
    print("\n" + "=" * 72)
    print(f"REVIEW ITEM: {summary['item_dir']}")
    print("=" * 72)
    print(f"prompt_version : {script['prompt_version']}")
    print(f"tiers (entries): {script['entry_count']}")
    print(f"word_count     : {script['word_count']}")
    print(f"\nCHOSEN TITLE   : {summary['chosen_title']}")
    print(f"CHOSEN HOOK    : {summary['chosen_hook']}")

    print("\nALL 5 TITLE VARIANTS:")
    for variant in packaging.build_title_variants(topic, entry_count=script["entry_count"]):
        print(f"  [{variant['formula']:16}] {variant['title']}")

    print("\nTHUMBNAILS:")
    for thumb in summary["thumbnails"]:
        print(f"  - {thumb['concept_id']:16} provider={thumb['provider']} "
              f"placeholder={thumb['is_placeholder']}")
        print(f"      {thumb['path']}")

    print("\nCOLD HOOK (chosen variant shown above; script cold_hook):")
    print(f"  {script['cold_hook']}")
    print("\nTIERS:")
    for tier in script["tiers"]:
        print(f"  T{tier['tier']} [{tier['label']}] {tier['entry_title']}")
        print(f"     narration : {tier['narration'][:160]}{'...' if len(tier['narration'])>160 else ''}")
        print(f"     shots     : {', '.join(tier['shot_list'])}")
        print(f"     open-loop : {tier['open_loop']}")
    print(f"\nFINAL PAYOFF: {script['final_payoff']}")
    print(f"\nFull package: {summary['package_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
