import json
import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import exemplars, style_spec


def _snip_text(n_words: int) -> list:
    """A transcript (list of snippets) with ~n_words words across timed snippets."""
    words = " ".join(f"w{i}" for i in range(n_words))
    return [{"text": words, "start": 0.0, "duration": 30.0},
            {"text": "more narration here about the topic", "start": 31.5, "duration": 5.0}]


def _row(vid, ch, views, vel=0.0, out=0.0, title="The Something Iceberg Explained", dur=3000):
    return {"video_id": vid, "channel_id": ch, "views": views, "view_velocity": vel,
            "outlier_score": out, "title": title, "duration_s": dur}


# --------------------------------------------------------------------------- #
# Selection: blended rank + relevance pre-harvest + anti-overfit cap
# --------------------------------------------------------------------------- #
class HarvestSelectionTests(unittest.TestCase):
    def test_blended_rank_orders_by_combined_metrics(self):
        rows = [_row("low", "c1", 100, 1, 1), _row("high", "c2", 1000, 10, 10)]
        ranked = exemplars._blended_rank(rows)
        self.assertEqual(ranked[0]["video_id"], "high")
        self.assertGreaterEqual(ranked[0]["blended"], ranked[1]["blended"])

    def test_relevance_filter_excludes_offtopic_before_harvest(self):
        rows = [
            _row("keep", "c1", 900, title="The Backrooms Iceberg Explained"),
            _row("drop", "c2", 800, title="Lost (TV series) full recap"),
        ]

        def fake_llm(prompt):
            # research.filter_relevant prompt -> mark index 1 off-topic.
            self.assertIn("ON-TOPIC", prompt)
            return ('[{"index": 0, "on_topic": true, "reason": "iceberg video"},'
                    ' {"index": 1, "on_topic": false, "reason": "a TV show, not the genre"}]')

        kept = exemplars.harvest_exemplars(
            "iceberg_deepdive", llm=fake_llm, rows=rows,
            transcript_fn=lambda vid: _snip_text(80),
        )
        ids = [e["video_id"] for e in kept]
        self.assertIn("keep", ids)
        self.assertNotIn("drop", ids)

    def test_per_channel_cap_prevents_one_creator_dominating(self):
        rows = [_row(f"v{i}", "samechannel", 1000 - i) for i in range(5)]
        kept = exemplars.harvest_exemplars(
            "iceberg_deepdive", llm=None, rows=rows, per_channel_cap=2, max_keep=10,
            transcript_fn=lambda vid: _snip_text(80),
        )
        self.assertEqual(len(kept), 2)  # capped at 2 from the single channel

    def test_transcript_missing_is_skipped_and_next_pulled(self):
        rows = [_row("notrans", "c1", 1000), _row("hastrans", "c2", 900)]

        def transcript_fn(vid):
            return None if vid == "notrans" else _snip_text(80)

        kept = exemplars.harvest_exemplars(
            "iceberg_deepdive", llm=None, rows=rows, transcript_fn=transcript_fn,
        )
        ids = [e["video_id"] for e in kept]
        self.assertEqual(ids, ["hastrans"])  # missing transcript skipped


# --------------------------------------------------------------------------- #
# Transcript cleaning + derivations
# --------------------------------------------------------------------------- #
class TranscriptCleaningTests(unittest.TestCase):
    def test_clean_strips_noise_and_adds_breaks(self):
        snips = [{"text": "hello [Music] world", "start": 0.0, "duration": 1.0},
                 {"text": "after a long pause", "start": 5.0, "duration": 1.0}]
        clean = exemplars.clean_transcript(snips)
        self.assertNotIn("[Music]", clean)
        self.assertIn("world.", clean)  # gap > 1.2s inserts a sentence break

    def test_parse_chapters_from_description(self):
        desc = "Intro\n0:00 The Surface\n2:15 The Deep\nnot a chapter line"
        chapters = exemplars.parse_chapters(desc)
        self.assertEqual([c["title"] for c in chapters], ["The Surface", "The Deep"])


# --------------------------------------------------------------------------- #
# Style distillation + parsing
# --------------------------------------------------------------------------- #
class StyleSpecParsingTests(unittest.TestCase):
    def _exemplars(self):
        return [
            {"video_id": "a", "channel_id": "c1", "title": "T1", "views": 100,
             "blended": 0.9, "transcript": "alpha bravo charlie delta echo foxtrot golf hotel",
             "structure": {"pacing": {"wpm": 150, "avg_sentence_words": 12}}},
            {"video_id": "b", "channel_id": "c2", "title": "T2", "views": 90,
             "blended": 0.8, "transcript": "completely different words over here entirely",
             "structure": {"pacing": {"wpm": 140, "avg_sentence_words": 10}}},
        ]

    def test_distill_parses_spec_fields(self):
        spec_json = json.dumps({
            "hook_formula": "open on a stake", "narrator_tone": "ominous",
            "pacing_notes": "short then long", "dread_build": "imply more",
            "tier_transitions": "open loop each tier", "words_per_entry": 175,
            "escalation_pattern": "darker each tier", "title_patterns": ["The {Topic} Iceberg"],
            "chapter_style": "tier: name", "illustrative_snippets": ["a short paraphrase"],
        })
        spec = style_spec.distill_style_spec("iceberg_deepdive", self._exemplars(), lambda p: spec_json)
        self.assertFalse(spec["low_confidence"])
        self.assertEqual(spec["words_per_entry"], 175)
        self.assertEqual(spec["distinct_channels"], 2)
        self.assertEqual(len(spec["learned_from"]), 2)
        self.assertEqual(spec["title_patterns"], ["The {Topic} Iceberg"])

    def test_distill_falls_back_to_default_on_bad_json(self):
        spec = style_spec.distill_style_spec("n", self._exemplars(), lambda p: "not json")
        self.assertTrue(spec["low_confidence"])

    def test_no_exemplars_returns_default(self):
        spec = style_spec.distill_style_spec("n", [], lambda p: "{}")
        self.assertTrue(spec["low_confidence"])
        self.assertEqual(spec["learned_from"], [])

    def test_build_caches_and_reloads_without_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row("a", "c1", 100), _row("b", "c2", 90)]
            calls = {"llm": 0}

            def fake_llm(prompt):
                calls["llm"] += 1
                if "ON-TOPIC" in prompt:  # relevance filter
                    return '[{"index":0,"on_topic":true},{"index":1,"on_topic":true}]'
                return json.dumps({"hook_formula": "x", "title_patterns": ["The {Topic} Iceberg"]})

            spec = style_spec.build_style_spec(
                "iceberg_deepdive", llm=fake_llm, refresh=True, rows=rows,
                transcript_fn=lambda vid: _snip_text(80), cache_dir=tmp,
            )
            self.assertFalse(spec["low_confidence"])
            self.assertTrue(os.path.exists(style_spec.cache_path("iceberg_deepdive", tmp)))
            # cached spec must NOT contain transcripts
            with open(style_spec.cache_path("iceberg_deepdive", tmp), encoding="utf-8") as h:
                self.assertNotIn("transcript", h.read())
            # second call (refresh=False) loads cache -> no further llm calls
            before = calls["llm"]
            spec2 = style_spec.build_style_spec("iceberg_deepdive", llm=fake_llm,
                                                refresh=False, cache_dir=tmp)
            self.assertEqual(calls["llm"], before)
            self.assertEqual(spec2["hook_formula"], spec["hook_formula"])


# --------------------------------------------------------------------------- #
# Legal guards: anti-plagiarism leak + snippet sanitization
# --------------------------------------------------------------------------- #
class LegalGuardTests(unittest.TestCase):
    EXEMPLAR = ("the ancient cults worshipped forgotten gods beneath the waves "
                "in silence for ten thousand years")

    def test_find_leak_detects_long_verbatim_run(self):
        generated = "Today we explore the ancient cults worshipped forgotten gods beneath the waves tonight."
        leak = style_spec.find_leak(generated, [self.EXEMPLAR], min_words=8)
        self.assertIsNotNone(leak)

    def test_find_leak_clears_original_writing(self):
        generated = "Our own grounded narration shares no long phrase with the source material at all."
        self.assertIsNone(style_spec.find_leak(generated, [self.EXEMPLAR], min_words=8))

    def test_assert_no_leak_raises_on_leak(self):
        with self.assertRaises(ValueError):
            style_spec.assert_no_leak(
                "the ancient cults worshipped forgotten gods beneath the waves now",
                [self.EXEMPLAR], min_words=8,
            )

    def test_sanitize_truncates_snippets_and_scrubs_leaks(self):
        spec = style_spec.default_spec("n")
        spec["illustrative_snippets"] = [" ".join(f"x{i}" for i in range(25))]  # 25 words
        spec["dread_build"] = "the ancient cults worshipped forgotten gods beneath the waves"  # verbatim leak
        cleaned = style_spec.sanitize_spec(spec, [self.EXEMPLAR])
        self.assertTrue(all(len(s.split()) <= style_spec.SNIPPET_MAX_WORDS
                            for s in cleaned["illustrative_snippets"]))
        # leaking field scrubbed back to the safe default
        self.assertNotIn("ancient cults worshipped", cleaned["dread_build"])

    def test_format_spec_for_prompt_empty_when_none(self):
        self.assertEqual(style_spec.format_spec_for_prompt(None), "")
        self.assertIn("Hook formula", style_spec.format_spec_for_prompt(style_spec.default_spec("n")))


if __name__ == "__main__":
    unittest.main()
