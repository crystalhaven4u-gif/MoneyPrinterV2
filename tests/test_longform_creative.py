import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import hooks, packaging, script, storage


# --------------------------------------------------------------------------- #
# Script parser (iceberg tiers)
# --------------------------------------------------------------------------- #
class IcebergScriptParserTests(unittest.TestCase):
    SAMPLE = """
    Sure, here is your script:
    ```json
    {
      "iceberg_topic": "Deep Sea",
      "cold_hook": "Welcome to the deep sea iceberg. By the bottom, something is watching.",
      "tiers": [
        {"tier": 1, "label": "Surface", "entry_title": "Bioluminescence",
         "narration": "Near the top, life still glows with three words here.",
         "shot_list": ["glowing jellyfish", "dark water"],
         "open_loop": "But it gets darker."},
        {"tier": 2, "label": "Abyss", "entry_title": "The Bloop",
         "narration": "Deeper still, a sound no one explained for years and years.",
         "shot_list": "sonar screen",
         "open_loop": "And below that, the worst."}
      ],
      "final_payoff": "At the very bottom waits the thing the hook promised."
    }
    ```
    Hope that helps!
    """

    def test_parses_tiers_and_counts(self):
        result = script.parse_iceberg_script(self.SAMPLE)
        self.assertEqual(result["entry_count"], 2)
        self.assertEqual(result["iceberg_topic"], "Deep Sea")
        self.assertTrue(result["cold_hook"])
        self.assertTrue(result["final_payoff"])
        self.assertGreater(result["word_count"], 0)

    def test_shot_list_coerced_and_open_loops_present(self):
        result = script.parse_iceberg_script(self.SAMPLE)
        # string shot_list is coerced to a list
        self.assertEqual(result["tiers"][1]["shot_list"], ["sonar screen"])
        for tier in result["tiers"]:
            self.assertTrue(tier["open_loop"])
            self.assertTrue(tier["entry_title"])

    def test_raises_without_tiers(self):
        with self.assertRaises(ValueError):
            script.parse_iceberg_script('{"cold_hook": "hi", "tiers": []}')

    def test_raises_without_json(self):
        with self.assertRaises(ValueError):
            script.parse_iceberg_script("no json at all here")

    def test_generate_script_uses_injected_llm_and_logs(self):
        # few-shot reads farm.db; isolate to a temp db so no real winners needed.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            calls = {"n": 0}

            def fake_llm(prompt):
                calls["n"] += 1
                # The prompt must carry the versioned instructions + topic.
                assert "TOPIC: Roman Empire" in prompt
                return self.SAMPLE

            result = script.generate_script(
                "Roman Empire", llm=fake_llm, target_minutes=10, db_path=db,
                run_id="run-1",
            )
            self.assertEqual(calls["n"], 1)
            self.assertEqual(result["prompt_version"], "iceberg-v1")
            runs = storage.get_creative_runs(db_path=db)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["entry_count"], 2)
            self.assertEqual(runs[0]["prompt_version"], "iceberg-v1")


# --------------------------------------------------------------------------- #
# Hook scorer
# --------------------------------------------------------------------------- #
class HookScorerTests(unittest.TestCase):
    def test_parse_hook_list_array_of_strings(self):
        out = hooks.parse_hook_list('["hook one", "hook two", "  "]')
        self.assertEqual(out, ["hook one", "hook two"])

    def test_parse_hook_list_array_of_objects(self):
        out = hooks.parse_hook_list('[{"hook": "a"}, {"hook": "b"}]')
        self.assertEqual(out, ["a", "b"])

    def test_parse_hook_list_falls_back_to_numbered_list(self):
        raw = (
            "Here are your hooks:\n"
            "1. The Lost Media iceberg goes deeper than anyone admits.\n"
            "2) At the bottom waits something nobody has ever found.\n"
            "- How deep does the Lost Media iceberg really go?\n"
        )
        out = hooks.parse_hook_list(raw)
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith("The Lost Media iceberg"))
        self.assertNotIn("Here are your hooks", out)

    def test_parse_scores_clamps_and_scales(self):
        # 0-1 inputs scale to 0-10; out-of-range clamps.
        scores = hooks.parse_scores('{"curiosity": 0.5, "depth_pull": 12, "payoff_promise": -3}')
        self.assertEqual(scores["curiosity"], 5.0)
        self.assertEqual(scores["depth_pull"], 10.0)
        self.assertEqual(scores["payoff_promise"], 0.0)

    def test_aggregate_score_weighting(self):
        total = hooks.aggregate_score(
            {"curiosity": 10, "depth_pull": 0, "payoff_promise": 0}
        )
        self.assertEqual(total, 4.0)  # 0.4 * 10

    def test_select_best_picks_highest_then_breaks_ties(self):
        scored = [
            {"hook": "low", "scores": {"curiosity": 1}, "total": 2.0},
            {"hook": "tie-b", "scores": {"curiosity": 5}, "total": 8.0},
            {"hook": "tie-a", "scores": {"curiosity": 5}, "total": 8.0},
        ]
        best = hooks.select_best(scored)
        # tie on total -> higher curiosity (both 5) -> hook text desc ("tie-b")
        self.assertEqual(best["hook"], "tie-b")

    def test_run_hook_engine_scores_and_logs_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")

            def fake_llm(prompt):
                if "Write" in prompt and "cold-open hooks" in prompt:
                    return '["The X iceberg hides something.", "How deep does X go?"]'
                # scoring call: make the second hook win
                if "How deep does X go?" in prompt:
                    return '{"curiosity": 9, "depth_pull": 9, "payoff_promise": 8}'
                return '{"curiosity": 3, "depth_pull": 3, "payoff_promise": 3}'

            result = hooks.run_hook_engine(
                "X", llm=fake_llm, n=6, db_path=db, run_id="run-h"
            )
            self.assertEqual(result["best"]["hook"], "How deep does X go?")
            rows = storage.get_hook_variants("run-h", db_path=db)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(r["is_best"] for r in rows), 1)


# --------------------------------------------------------------------------- #
# Titles + thumbnail concepts
# --------------------------------------------------------------------------- #
class TitleAndConceptTests(unittest.TestCase):
    def test_five_titles_with_required_formulas(self):
        titles = packaging.build_title_variants("internet mysteries", entry_count=8)
        self.assertEqual(len(titles), 5)
        formulas = {t["formula"] for t in titles}
        self.assertIn("iceberg", formulas)
        self.assertIn("specific_number", formulas)
        self.assertIn("curiosity_gap", formulas)
        # topic title-cased and present in every title
        for t in titles:
            self.assertIn("Internet Mysteries", t["title"])

    def test_specific_number_is_odd(self):
        titles = packaging.build_title_variants("x", entry_count=8)
        num_title = [t for t in titles if t["formula"] == "specific_number"][0]["title"]
        leading = int(num_title.split()[0])
        self.assertEqual(leading % 2, 1)

    def test_three_thumbnail_concepts(self):
        concepts = packaging.build_thumbnail_concepts("deep sea")
        self.assertEqual(len(concepts), 3)
        for concept in concepts:
            self.assertTrue(concept["visual_prompt"])
            self.assertTrue(concept["overlay_text"])
            self.assertEqual(len({c["concept_id"] for c in concepts}), 3)


# --------------------------------------------------------------------------- #
# Packaging calls generate_image() (mocked) -- no real network
# --------------------------------------------------------------------------- #
class FakeImageResult:
    def __init__(self, path):
        self.path = path
        self.provider = "pollinations"
        self.is_placeholder = False


class PackagingRenderTests(unittest.TestCase):
    def test_render_thumbnail_calls_generate_image_with_16x9(self):
        captured = {}

        def fake_generate_image(**kwargs):
            captured.update(kwargs)
            return FakeImageResult(kwargs["output_path"])

        concept = packaging.build_thumbnail_concepts("x")[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "thumb.png")
            result = packaging.render_thumbnail(
                concept, out, generate_image_fn=fake_generate_image, do_overlay=False
            )

        self.assertEqual(captured["aspect_ratio"], "16:9")
        self.assertEqual(captured["prompt"], concept["visual_prompt"])
        self.assertEqual(result["provider"], "pollinations")
        self.assertFalse(result["is_placeholder"])

    def test_submit_to_review_writes_item_and_logs_choices(self):
        calls = {"n": 0}

        def fake_generate_image(**kwargs):
            calls["n"] += 1
            return FakeImageResult(kwargs["output_path"])

        topic = "deep sea"
        script_obj = {
            "iceberg_topic": topic,
            "cold_hook": "hook",
            "tiers": [{"tier": 1, "label": "Surface", "entry_title": "e",
                       "narration": "n", "shot_list": ["s"], "open_loop": "o"}],
            "final_payoff": "p",
            "entry_count": 1,
            "word_count": 4,
            "prompt_version": "iceberg-v1",
        }
        hooks_result = {
            "best": {"hook": "best hook", "scores": {}, "total": 9.0},
            "variants": [{"hook": "best hook", "scores": {}, "total": 9.0}],
        }
        titles = packaging.build_title_variants(topic, entry_count=1)

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            base = os.path.join(tmp, "review")
            # seed the run row so the choices UPDATE has a target
            storage.log_creative_run(
                {"run_id": "run-p", "created_at": "t", "niche": "iceberg_deepdive",
                 "topic": topic, "prompt_version": "iceberg-v1", "entry_count": 1,
                 "word_count": 4, "chosen_title": None, "chosen_hook": None,
                 "review_item_path": None},
                db_path=db,
            )
            summary = packaging.submit_to_review(
                run_id="run-p",
                topic=topic,
                script=script_obj,
                hooks_result=hooks_result,
                titles=titles,
                concepts=packaging.build_thumbnail_concepts(topic),
                base_dir=base,
                generate_image_fn=fake_generate_image,
                do_overlay=False,
                db_path=db,
            )

            self.assertEqual(calls["n"], 3)  # one generate_image per concept
            self.assertTrue(os.path.exists(summary["package_path"]))
            self.assertEqual(summary["chosen_hook"], "best hook")
            self.assertEqual(summary["chosen_title"], titles[0]["title"])

            runs = storage.get_creative_runs(db_path=db)
            self.assertEqual(runs[0]["chosen_title"], titles[0]["title"])
            self.assertEqual(runs[0]["chosen_hook"], "best hook")
            title_rows = storage.get_title_variants("run-p", db_path=db)
            self.assertEqual(len(title_rows), 5)
            self.assertEqual(sum(r["is_chosen"] for r in title_rows), 1)

    def test_no_real_image_provider_network(self):
        # Guard: with a mocked generate_image_fn, the real image_providers.requests
        # is never touched.
        with patch.object(packaging.image_providers, "requests") as fake_requests:
            fake_requests.request.side_effect = AssertionError("real network used!")

            def fake_generate_image(**kwargs):
                return FakeImageResult(kwargs["output_path"])

            concept = packaging.build_thumbnail_concepts("x")[0]
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "t.png")
                packaging.render_thumbnail(
                    concept, out, generate_image_fn=fake_generate_image, do_overlay=False
                )
            fake_requests.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
