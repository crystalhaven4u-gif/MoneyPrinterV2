import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import compose, music, sourcer, storage, thumbnail, tts


def _touch(path, data=b"x"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


# --------------------------------------------------------------------------- #
# Commercial-license filter
# --------------------------------------------------------------------------- #
class LicenseFilterTests(unittest.TestCase):
    def test_commercial_safe_accepts_and_rejects(self):
        self.assertTrue(sourcer.is_commercial_safe("Pexels License"))
        self.assertTrue(sourcer.is_commercial_safe("cc0"))
        self.assertTrue(sourcer.is_commercial_safe("CC BY 4.0"))
        self.assertFalse(sourcer.is_commercial_safe("CC BY-NC 4.0"))
        self.assertFalse(sourcer.is_commercial_safe("CC BY-ND"))
        self.assertFalse(sourcer.is_commercial_safe(""))


# --------------------------------------------------------------------------- #
# Sourcer: license logging + dedupe + AI fallback
# --------------------------------------------------------------------------- #
class SourcerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dest = self._tmp.name
        self.db = os.path.join(self.dest, "farm.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _provider(self, source_id="vid1"):
        def provider(shot, cfg, session):
            return [{
                "source": "pexels", "source_id": source_id, "url": "http://pexels/v",
                "download_url": "http://pexels/v.mp4", "license": "Pexels License",
                "attribution_required": False, "kind": "video", "author": "Jane",
            }]
        return provider

    def test_safe_clip_is_logged_with_license(self):
        with patch.object(sourcer, "download", side_effect=lambda url, dest, session=None: _touch(dest)), \
             patch.object(sourcer, "perceptual_hash", return_value="HASH-A"):
            asset = sourcer.source_shot(
                "empty room", 0, "Surface", "run-x", self.dest,
                providers=[self._provider()], used_hashes=set(), db_path=self.db,
            )
        self.assertEqual(asset["source"], "pexels")
        self.assertEqual(asset["license"], "Pexels License")
        rows = storage.get_assets("run-x", db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["license"])

    def test_perceptual_dedupe_skips_repeat_then_uses_ai(self):
        used = set()
        ai_file = _touch(os.path.join(self.dest, "ai.png"))

        class _Img:
            path = ai_file
            is_placeholder = False

        with patch.object(sourcer, "download", side_effect=lambda url, dest, session=None: _touch(dest)), \
             patch.object(sourcer, "perceptual_hash", return_value="SAME"):
            first = sourcer.source_shot(
                "room", 0, "S", "run-d", self.dest,
                providers=[self._provider("v1")], used_hashes=used, db_path=self.db,
            )
            # second shot: same provider + identical hash -> dup -> AI fallback
            second = sourcer.source_shot(
                "room again", 1, "S", "run-d", self.dest,
                providers=[self._provider("v1")], used_hashes=used,
                image_fallback_fn=lambda prompt, out: _Img(), db_path=self.db,
            )
        self.assertEqual(first["source"], "pexels")
        self.assertEqual(second["source"], "ai_generated")
        self.assertEqual(second["kind"], "image")

    def test_ai_fallback_when_no_providers(self):
        ai_file = _touch(os.path.join(self.dest, "ai.png"))

        class _Img:
            path = ai_file
            is_placeholder = False

        asset = sourcer.source_shot(
            "shot", 0, "S", "run-ai", self.dest, providers=[],
            image_fallback_fn=lambda prompt, out: _Img(), db_path=self.db,
        )
        self.assertEqual(asset["source"], "ai_generated")
        self.assertEqual(asset["license"], "AI-generated (owned)")

    def test_slate_last_resort_still_licensed(self):
        asset = sourcer.source_shot(
            "shot", 0, "S", "run-slate", self.dest, providers=[],
            image_fallback_fn=None, db_path=self.db,
        )
        self.assertEqual(asset["source"], "slate")
        self.assertTrue(asset["license"])
        self.assertTrue(os.path.exists(asset["path"]))


# --------------------------------------------------------------------------- #
# Relevance ranking + AI-primary / placeholder-fallthrough sourcing
# --------------------------------------------------------------------------- #
class RelevanceRankerTests(unittest.TestCase):
    def _cands(self):
        return [
            {"source": "pixabay", "source_id": "a", "kind": "video",
             "download_url": "u/a", "license": "Pixabay License", "description": "city traffic day"},
            {"source": "pixabay", "source_id": "b", "kind": "video",
             "download_url": "u/b", "license": "Pixabay License", "description": "dark empty liminal hallway"},
        ]

    def test_rank_orders_best_first_and_reports_best_score(self):
        # ranker likes candidate index 1 (the liminal hallway) far more.
        def ranker(intent, descriptions):
            return [2.0, 9.0]

        ordered, best = sourcer.rank_candidates("eerie liminal hallway", self._cands(), ranker)
        self.assertEqual(ordered[0]["source_id"], "b")
        self.assertEqual(best, 9.0)

    def test_no_ranker_keeps_provider_order(self):
        ordered, best = sourcer.rank_candidates("x", self._cands(), None)
        self.assertEqual([c["source_id"] for c in ordered], ["a", "b"])
        self.assertIsNone(best)

    def test_make_relevance_ranker_parses_llm_scores(self):
        def fake_llm(prompt):
            return '[{"index": 0, "score": 3}, {"index": 1, "score": 8}]'

        ranker = sourcer.make_relevance_ranker(fake_llm)
        self.assertEqual(ranker("intent", ["d0", "d1"]), [3.0, 8.0])

    def test_make_relevance_ranker_nonblocking_on_error(self):
        def boom(prompt):
            raise RuntimeError("down")

        ranker = sourcer.make_relevance_ranker(boom)
        self.assertEqual(ranker("intent", ["d0", "d1"]), [])

    def test_source_shot_picks_best_ranked_candidate(self):
        def provider(shot, cfg, session):
            return [
                {"source": "pixabay", "source_id": "lowmatch", "kind": "video",
                 "download_url": "u/low", "license": "Pixabay License", "description": "sunny beach"},
                {"source": "pixabay", "source_id": "bestmatch", "kind": "video",
                 "download_url": "u/best", "license": "Pixabay License", "description": "dark hallway"},
            ]

        ranker = lambda intent, descs: [1.0, 9.0]  # second is the winner
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            with patch.object(sourcer, "download", side_effect=lambda url, dest, session=None: _touch(dest)), \
                 patch.object(sourcer, "perceptual_hash", side_effect=lambda *a, **k: object()):
                asset = sourcer.source_shot(
                    "hallway", 0, "S", "run-rank", tmp, providers=[provider],
                    used_hashes=set(), ranker=ranker, db_path=db,
                )
        self.assertEqual(asset["source_id"], "bestmatch")

    def test_generic_mismatch_falls_through_to_ai(self):
        def provider(shot, cfg, session):
            return [{"source": "pixabay", "source_id": "generic", "kind": "video",
                     "download_url": "u/g", "license": "Pixabay License", "description": "random stock"}]

        ranker = lambda intent, descs: [1.0]  # below MIN_RELEVANCE -> reject

        class _Img:
            is_placeholder = False

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            ai_file = _touch(os.path.join(tmp, "ai.png"))
            _Img.path = ai_file
            with patch.object(sourcer, "download", side_effect=lambda url, dest, session=None: _touch(dest)), \
                 patch.object(sourcer, "perceptual_hash", side_effect=lambda *a, **k: object()):
                asset = sourcer.source_shot(
                    "hallway", 0, "S", "run-gen", tmp, providers=[provider],
                    used_hashes=set(), ranker=ranker,
                    image_fallback_fn=lambda prompt, out: _Img(), db_path=db,
                )
        self.assertEqual(asset["source"], "ai_generated")

    def test_ai_placeholder_falls_through_to_stock(self):
        # prefer_ai, but AI returns a flagged placeholder -> must use stock, not slate.
        def provider(shot, cfg, session):
            return [{"source": "pixabay", "source_id": "real", "kind": "video",
                     "download_url": "u/r", "license": "Pixabay License", "description": "dark hallway"}]

        class _Placeholder:
            path = None
            is_placeholder = True

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            with patch.object(sourcer, "download", side_effect=lambda url, dest, session=None: _touch(dest)), \
                 patch.object(sourcer, "perceptual_hash", side_effect=lambda *a, **k: object()):
                asset = sourcer.source_shot(
                    "hallway", 0, "S", "run-ph", tmp, providers=[provider],
                    used_hashes=set(), prefer_ai=True,
                    image_fallback_fn=lambda prompt, out: _Placeholder(), db_path=db,
                )
        self.assertEqual(asset["source"], "pixabay")  # not slate


# --------------------------------------------------------------------------- #
# Iceberg thumbnail generator
# --------------------------------------------------------------------------- #
class IcebergThumbnailTests(unittest.TestCase):
    def test_uses_ai_base_when_not_placeholder(self):
        class _Img:
            is_placeholder = False

        def fake_gen(**kwargs):
            _Img.path = _touch(kwargs["output_path"])
            return _Img()

        called = {"procedural": 0}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(thumbnail, "procedural_base",
                              side_effect=lambda *a, **k: called.__setitem__("procedural", called["procedural"] + 1)):
                base = thumbnail.generate_base(
                    "X", thumbnail.VARIANTS[0], os.path.join(tmp, "b.png"), "magick",
                    generate_image_fn=fake_gen,
                )
        self.assertEqual(base["source"], "ai_generated")
        self.assertEqual(called["procedural"], 0)  # AI worked -> no procedural draw

    def test_falls_back_to_procedural_on_placeholder(self):
        class _Img:
            path = None
            is_placeholder = True

        def fake_gen(**kwargs):
            return _Img()

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "b.png")
            with patch.object(thumbnail, "procedural_base",
                              side_effect=lambda out, magick, palette: _touch(out)):
                base = thumbnail.generate_base(
                    "X", thumbnail.VARIANTS[0], target, "magick", generate_image_fn=fake_gen,
                )
        self.assertEqual(base["source"], "procedural")

    def test_title_lines_strip_leading_the(self):
        self.assertEqual(thumbnail._title_lines("The Backrooms"), "THE BACKROOMS\nICEBERG")


# --------------------------------------------------------------------------- #
# License-manifest completeness
# --------------------------------------------------------------------------- #
class ManifestTests(unittest.TestCase):
    def test_manifest_complete_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            storage.log_asset({"run_id": "r1", "license": "Pexels License",
                               "source": "pexels", "shot_index": 0}, db_path=db)
            self.assertTrue(storage.license_manifest("r1", db_path=db)["complete"])

            storage.log_asset({"run_id": "r2", "license": "Pexels License",
                               "source": "pexels", "shot_index": 0}, db_path=db)
            storage.log_asset({"run_id": "r2", "license": "", "source": "mystery",
                               "shot_index": 1}, db_path=db)
            manifest = storage.license_manifest("r2", db_path=db)
            self.assertFalse(manifest["complete"])
            self.assertEqual(len(manifest["unlicensed"]), 1)


# --------------------------------------------------------------------------- #
# Caption-timing mapping
# --------------------------------------------------------------------------- #
class CaptionTimingTests(unittest.TestCase):
    def test_chunks_respect_word_and_time_limits(self):
        timings = [{"word": f"w{i}", "start": i * 1.0, "end": i * 1.0 + 0.9} for i in range(10)]
        chunks = compose.build_caption_chunks(timings, max_words=3, max_secs=10)
        self.assertTrue(all(len(c["text"].split()) <= 3 for c in chunks))
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["text"], "w0 w1 w2")
        # full coverage: last chunk ends at last word's end
        self.assertAlmostEqual(chunks[-1]["end"], timings[-1]["end"])

    def test_time_limit_splits(self):
        timings = [{"word": "a", "start": 0.0, "end": 1.0},
                   {"word": "b", "start": 1.0, "end": 5.0}]  # span > max_secs
        chunks = compose.build_caption_chunks(timings, max_words=10, max_secs=3.0)
        self.assertEqual(len(chunks), 2)

    def test_chapters_cumulative(self):
        chapters = compose.build_chapters([
            {"title": "Intro", "duration": 10}, {"title": "Tier 1", "duration": 20},
        ])
        self.assertEqual(chapters[0]["start"], 0.0)
        self.assertEqual(chapters[1]["start"], 10.0)
        self.assertEqual(chapters[1]["end"], 30.0)
        text = compose.format_chapters_for_description(chapters)
        self.assertTrue(text.startswith("0:00 Intro"))

    def test_credits_dedupe_attribution_required_only(self):
        assets = [
            {"attribution_required": True, "attribution": "Jane via openverse (CC BY)"},
            {"attribution_required": True, "attribution": "Jane via openverse (CC BY)"},
            {"attribution_required": False, "attribution": ""},
        ]
        self.assertEqual(compose.build_credits(assets), ["Jane via openverse (CC BY)"])


# --------------------------------------------------------------------------- #
# Music bed fetch (no real network)
# --------------------------------------------------------------------------- #
class MusicTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dest = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _provider(self, license_str="cc-by", title="Dark Drone"):
        def provider(mood, cfg, session):
            return [{"source": "openverse", "source_id": "aud1",
                     "url": "http://ov/a", "download_url": "http://ov/a.mp3",
                     "license": license_str, "attribution_required": "by" in license_str,
                     "author": "Composer X", "title": title}]
        return provider

    def test_fetch_logs_license_and_caches(self):
        calls = {"n": 0}

        def fake_download(url, dest, session=None):
            calls["n"] += 1
            return _touch(dest)

        with patch.object(music.sourcer, "download", side_effect=fake_download):
            rec = music.fetch_music_bed("dark ambient", self.dest,
                                        providers=[self._provider("cc-by")])
            self.assertIsNotNone(rec)
            self.assertEqual(rec["source"], "openverse")
            self.assertTrue(rec["attribution_required"])
            self.assertIn("Dark Drone", rec["attribution"])
            # second call hits cache -> no extra download
            rec2 = music.fetch_music_bed("dark ambient", self.dest,
                                         providers=[self._provider("cc-by")])
            self.assertEqual(rec2["source"], "openverse")
            self.assertEqual(calls["n"], 1)  # cached

    def test_noncommercial_rejected_then_none(self):
        with patch.object(music.sourcer, "download", side_effect=lambda u, d, session=None: _touch(d)):
            rec = music.fetch_music_bed("x", self.dest,
                                        providers=[self._provider("cc-by-nc")], cache=False)
        self.assertIsNone(rec)  # NC filtered out, no other provider

    def test_provider_error_returns_none(self):
        def boom(mood, cfg, session):
            raise RuntimeError("down")

        rec = music.fetch_music_bed("x", self.dest, providers=[boom], cache=False)
        self.assertIsNone(rec)


# --------------------------------------------------------------------------- #
# TTS fallback (no real synthesis)
# --------------------------------------------------------------------------- #
class TTSFallbackTests(unittest.TestCase):
    def test_injected_synth_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "vo.mp3")

            def fake_synth(text, out_path, voice):
                _touch(out_path)
                return {"audio_path": out_path, "word_timings": [{"word": "hi", "start": 0, "end": 0.5}]}

            result = tts.synthesize_section("hello world", out, cfg={"provider": "edge_tts"}, synth=fake_synth)
            self.assertEqual(result["provider"], "injected")
            self.assertEqual(len(result["word_timings"]), 1)

    def test_failure_falls_back_to_silent_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "vo.mp3")

            def boom(text, out_path, voice):
                raise RuntimeError("no network")

            result = tts.synthesize_section("one two three four", out,
                                            cfg={"provider": "edge_tts"}, synth=boom)
            self.assertEqual(result["provider"], "estimated_silent")
            self.assertIsNone(result["audio_path"])
            self.assertEqual(len(result["word_timings"]), 4)  # estimated per word


if __name__ == "__main__":
    unittest.main()
