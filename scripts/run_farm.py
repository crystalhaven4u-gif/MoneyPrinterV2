#!/usr/bin/env python3
"""
Run a live YouTube long-form data farm from the repo root.

    python scripts/run_farm.py [--budget N] [--max-results N] [--recency-days N]

Reads the Data API key from longform.youtube_api_key in config.json, or the
YOUTUBE_API_KEY environment variable (which takes precedence). Writes
topics.longform.json and upserts rows into .mp/farm.db.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from longform import farmer

if __name__ == "__main__":
    raise SystemExit(farmer.main())
