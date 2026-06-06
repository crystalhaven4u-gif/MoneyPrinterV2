#!/usr/bin/env python3
"""
Readable report over the farmed YouTube data (.mp/farm.db).

Prints:
  - top videos per niche (views / view_velocity / outlier_score / score)
  - niche ranking by aggregate (mean) score
  - dominant title formulas observed in the real high performers

Usage:
    python scripts/farm_report.py [--top 5] [--db PATH]
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from longform import storage, title_match


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report on farmed long-form data.")
    parser.add_argument("--top", type=int, default=5, help="Top videos per niche to show.")
    parser.add_argument("--db", default=None, help="Override farm.db path.")
    args = parser.parse_args(argv)

    rows = storage.get_all(db_path=args.db)
    if not rows:
        print("No farmed data yet. Run a farm first (see README / farmer.main).")
        return 0

    by_niche = {}
    for row in rows:
        by_niche.setdefault(row["niche"], []).append(row)

    # Niche ranking by aggregate (mean) score.
    niche_aggregate = {
        niche: sum(r["score"] for r in videos) / len(videos)
        for niche, videos in by_niche.items()
    }
    ranked_niches = sorted(niche_aggregate.items(), key=lambda kv: kv[1], reverse=True)

    print("=" * 72)
    print(f"LONG-FORM FARM REPORT  —  {len(rows)} videos across {len(by_niche)} niches")
    print("=" * 72)

    print("\nNICHE RANKING (by mean score)")
    print("-" * 72)
    for rank, (niche, score) in enumerate(ranked_niches, start=1):
        videos = by_niche[niche]
        cc = videos[0].get("cc_feasibility", 0)
        print(
            f"{rank:>2}. {niche:<22} mean_score={score:.3f}  "
            f"videos={len(videos):<3} cc_feasibility={_fmt_int(cc)}"
        )

    print("\nTOP VIDEOS PER NICHE")
    print("-" * 72)
    for niche, _score in ranked_niches:
        videos = sorted(by_niche[niche], key=lambda r: r["score"], reverse=True)
        print(f"\n[{niche}]")
        for video in videos[: args.top]:
            print(f"  • {video['title'][:64]}")
            print(
                f"      score={video['score']:.3f}  views={_fmt_int(video['views'])}  "
                f"velocity={_fmt_int(round(video['view_velocity']))}/day  "
                f"outlier={video['outlier_score']:.2f}  "
                f"dur={round(video['duration_s'] / 60)}min"
            )
            print(f"      https://www.youtube.com/watch?v={video['video_id']}")

    print("\nDOMINANT TITLE FORMULAS (observed in real winners)")
    print("-" * 72)
    per_niche = title_match.analyze_by_niche(rows)
    for niche, _score in ranked_niches:
        ranking = per_niche.get(niche, [])
        if not ranking:
            print(f"[{niche}] (no formula matches)")
            continue
        formula_summary = ", ".join(f"{fid}×{count}" for fid, count in ranking[:4])
        print(f"[{niche}] {formula_summary}")

    # Overall formula tally.
    overall = title_match.tally_titles([r["title"] for r in rows])
    print("\nOVERALL FORMULA TALLY")
    print("-" * 72)
    for fid, count in overall["ranking"]:
        print(f"  {fid:<22} {count}")
    if overall["unmatched_formula_ids"]:
        print(f"\n(no heuristic for: {', '.join(overall['unmatched_formula_ids'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
