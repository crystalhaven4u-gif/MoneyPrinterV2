import json
import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import factcheck, packaging, script, storage


def _narration(n_words: int) -> str:
    return " ".join(f"w{i}" for i in range(n_words)) + "."


# --------------------------------------------------------------------------- #
# Specific extraction + new-entity diff (the lock primitive)
# --------------------------------------------------------------------------- #
class ExtractionTests(unittest.TestCase):
    def test_extracts_dates_numbers_names_skips_sentence_initial(self):
        text = "Imagine the dread. The Async Research Institute formed in 2019 with 7 members."
        spec = factcheck.extract_specifics(text, skip_initial=True)
        self.assertIn("2019", spec["dates"])
        self.assertIn("7", spec["numbers"])
        self.assertIn("async research institute", spec["proper_nouns"])
        self.assertNotIn("imagine", spec["proper_nouns"])  # sentence-initial style word

    def test_new_specifics_flags_invented_name_date_number(self):
        allowed = factcheck.allowed_specifics(
            "The Backrooms began on 4chan. Kane Parsons made a film."
        )
        candidate = ("That's what happened to Mark on February 10, 2022, "
                     "one of 5 witnesses Kane Parsons interviewed.")
        new = factcheck.new_specifics(allowed, candidate)
        self.assertIn("mark", new["proper_nouns"])
        self.assertIn("february 10, 2022", new["dates"])
        self.assertIn("5", new["numbers"])
        # Kane Parsons WAS in Pass 1 -> not flagged
        self.assertNotIn("kane parsons", new["proper_nouns"])

    def test_strip_sentences_removes_offending(self):
        text = "You descend into the dark. That's what happened to Mark. The dread grows."
        out = factcheck.strip_sentences_with(text, ["mark"])
        self.assertNotIn("Mark", out)
        self.assertIn("dread grows", out)


# --------------------------------------------------------------------------- #
# Pass 2 entity lock (integration via narrative_rewrite)
# --------------------------------------------------------------------------- #
class EntityLockTests(unittest.TestCase):
    def _script(self):
        return {
            "iceberg_topic": "The Backrooms",
            "cold_hook": "The Backrooms began as a 4chan thread.",
            "tiers": [{"tier": 1, "label": "Surface", "entry_title": "Creepypasta",
                       "narration": "The Backrooms are empty rooms from a 4chan post.",
                       "open_loop": "but it goes deeper"}],
            "final_payoff": "And that is the bottom.",
        }

    def test_invented_entities_stripped_when_reprompt_fails(self):
        # A Pass 2 that keeps injecting a new name + date; re-prompt can't fix it,
        # so the lock hard-strips the offending sentence and flags it.
        def stubborn_llm(prompt):
            return json.dumps({"narration":
                "You enter the rooms. That's what happened to Mark on March 3, 1999."})

        sc = script.narrative_rewrite(self._script(), stubborn_llm, prompt_cfg={})
        lock = sc["pass2_entity_lock"]
        self.assertTrue(lock["clean"])              # no new entity survives
        self.assertEqual(lock["new_entities_final"], 0)
        cold = sc["cold_hook"].lower()
        self.assertNotIn("mark", cold)
        self.assertNotIn("1999", cold)
        # the lock recorded the invented items it caught
        caught = lock["sections"][0]["invented_initial"]
        self.assertTrue(any("mark" in c for c in caught))

    def test_reprompt_recovers_clean_rewrite(self):
        calls = {"n": 0}

        def llm(prompt):
            calls["n"] += 1
            if "INTRODUCED specifics" in prompt:  # the relock re-prompt
                return json.dumps({"narration": "You enter the endless empty rooms, and the dread grows."})
            return json.dumps({"narration": "You meet Mark in the rooms."})  # first try invents Mark

        sc = script.narrative_rewrite(self._script(), llm, prompt_cfg={})
        self.assertTrue(sc["pass2_entity_lock"]["clean"])
        self.assertNotIn("Mark", sc["cold_hook"])
        self.assertGreaterEqual(sc["pass2_entity_lock"]["sections"][0]["retries"], 1)


# --------------------------------------------------------------------------- #
# Pass 1 verification (claim support against source)
# --------------------------------------------------------------------------- #
class VerificationTests(unittest.TestCase):
    SOURCE = ("The Backrooms originated on 4chan in 2019. The 2026 film was "
              "directed by Kane Parsons and stars Mark Duplass.")

    def test_supported_claims_pass(self):
        narration = "Born on 4chan in 2019, the 2026 film came from Kane Parsons."
        result = factcheck.verify_claims(narration, self.SOURCE)
        self.assertEqual(result["flagged"], [])
        self.assertEqual(result["score"], 1.0)

    def test_unsupported_date_and_name_flagged(self):
        narration = "Dr. Emma Taylor vanished in 2022, one of 9 researchers."
        result = factcheck.verify_claims(narration, self.SOURCE)
        kinds = {(f["type"], f["claim"]) for f in result["flagged"]}
        self.assertIn(("date", "2022"), kinds)
        self.assertIn(("number", "9"), kinds)
        self.assertTrue(any(f["type"] == "name" and "emma" in f["claim"] for f in result["flagged"]))

    def test_common_knowledge_not_flagged(self):
        result = factcheck.verify_claims("It spread across 4chan and Reddit.", "")
        self.assertEqual(result["flagged"], [])


# --------------------------------------------------------------------------- #
# Factual-confidence report is written into the review item (package.json)
# --------------------------------------------------------------------------- #
class ConfidenceReportInPackageTests(unittest.TestCase):
    def test_report_persisted_to_package_json(self):
        outline = json.dumps({
            "cold_hook": "The Backrooms began on 4chan.",
            "tiers": [{"tier": 1, "label": "Surface", "entry_title": "Creepypasta",
                       "premise": "p", "open_loop": "o"}],
            "final_payoff": "The end.",
        })
        # Pass-1 entry invents an unsupported date (1999) the source won't back.
        entry = json.dumps({"narration": "The rooms appeared in 1999. " + _narration(40),
                            "shot_list": ["empty room"], "open_loop": "deeper"})

        def fake_llm(prompt):
            if "Build the OUTLINE" in prompt:
                return outline
            if "INTRODUCED specifics" in prompt:  # relock (shouldn't add new here)
                return json.dumps({"narration": "You drift through the rooms as dread grows."})
            return entry  # entry generation + any rewrite

        def searcher(topic, n):
            return [{"title": "Creepypasta", "fact": "The Backrooms began on 4chan.",
                     "url": "http://w/backrooms", "source_text": "The Backrooms began on 4chan."}]

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "farm.db")
            sc = script.generate_script(
                "The Backrooms", llm=fake_llm, target_minutes=1, db_path=db,
                run_id="run-fc", grounding=True, searcher=searcher, narrative=True,
                style_spec={}, enrich=False,
            )
            self.assertIn("factual_confidence", sc)
            self.assertIn("flagged", sc["factual_confidence"])

            # submit_to_review embeds the script (with the report) in package.json
            base = os.path.join(tmp, "review")
            storage.log_creative_run(
                {"run_id": "run-fc", "created_at": "t", "niche": "iceberg_deepdive",
                 "topic": "The Backrooms", "prompt_version": sc["prompt_version"],
                 "entry_count": sc["entry_count"], "word_count": sc["word_count"],
                 "chosen_title": None, "chosen_hook": None, "review_item_path": None},
                db_path=db,
            )
            summary = packaging.submit_to_review(
                run_id="run-fc", topic="The Backrooms", script=sc,
                hooks_result={"best": {"hook": "h"}, "variants": []},
                titles=packaging.build_title_variants("The Backrooms", 1),
                concepts=[], base_dir=base, render=False, db_path=db,
            )
            with open(summary["package_path"], encoding="utf-8") as handle:
                package = json.load(handle)
            self.assertIn("factual_confidence", package["script"])
            self.assertIn("pass2_entity_lock", package["script"])


if __name__ == "__main__":
    unittest.main()
