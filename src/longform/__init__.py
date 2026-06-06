"""
Standalone faceless long-form YouTube engine.

This package is self-contained: it depends only on ``requests`` and the Python
standard library, and never imports the Shorts pipeline. The data-farming layer
(this first session) replaces research assumptions with real YouTube data.

Modules:
    youtube_api  - quota-aware YouTube Data API v3 client
    storage      - SQLite .mp/farm.db persistence (idempotent upserts)
    farmer       - farms candidate niche probes into farm.db + topics slate
    title_match  - matches farmed winners against the title-formula bank
"""
