import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import title_match


class MatchTitleTests(unittest.TestCase):
    def test_examples_match_expected_formulas(self):
        cases = {
            "The Man Who Saved The World": "the_man_who",
            "The Entire History of Rome, Explained": "entire_history",
            "13 Unsolved Mysteries Nobody Can Explain": "specific_number",
            "The Internet Mysteries Iceberg Explained": "iceberg",
            "What Actually Happened to Flight 370?": "provocative_question",
            "Why Everything You Know About the Pyramids Is Wrong": "contrarian",
            "Stop Using This Strategy in 2026 (It's Killing Your Channel)": "warning",
            "Former Detective Explains the Case That Was Never Solved": "authority_lead",
        }
        for title, expected in cases.items():
            self.assertIn(expected, title_match.match_title(title), msg=title)

    def test_title_can_match_multiple_formulas(self):
        matches = title_match.match_title("Tesla vs Edison: Who Really Won?")
        self.assertIn("versus", matches)
        self.assertIn("provocative_question", matches)  # ends with "?"

    def test_compression_and_specific_number_both_match(self):
        matches = title_match.match_title("30 Years of Business Knowledge in 2hrs 26mins")
        self.assertIn("compression", matches)
        self.assertIn("specific_number", matches)

    def test_no_match_returns_empty(self):
        self.assertEqual(title_match.match_title("My Thoughts On Stuff"), [])
        self.assertEqual(title_match.match_title(""), [])


class TallyTests(unittest.TestCase):
    def test_tally_counts_and_ranking(self):
        titles = [
            "The Man Who Saved The World",        # the_man_who
            "The Man Who Built Rome",             # the_man_who
            "The Internet Iceberg Explained",     # iceberg
            "My Thoughts On Stuff",               # no match
        ]
        result = title_match.tally_titles(titles)

        self.assertEqual(result["counts"]["the_man_who"], 2)
        self.assertEqual(result["counts"]["iceberg"], 1)
        self.assertEqual(result["ranking"][0], ("the_man_who", 2))
        self.assertEqual(result["matched_titles"], 3)
        self.assertEqual(result["total_titles"], 4)

    def test_every_bank_formula_has_a_heuristic(self):
        # Guards against a formula being added to the bank with no detector.
        self.assertEqual(title_match.tally_titles([])["unmatched_formula_ids"], [])

    def test_analyze_by_niche(self):
        rows = [
            {"niche": "history_doc", "title": "The Man Who Saved The World"},
            {"niche": "history_doc", "title": "The Man Who Built Rome"},
            {"niche": "iceberg_deepdive", "title": "The AI Iceberg Explained"},
        ]
        per_niche = title_match.analyze_by_niche(rows)
        self.assertEqual(per_niche["history_doc"][0], ("the_man_who", 2))
        self.assertEqual(per_niche["iceberg_deepdive"][0], ("iceberg", 1))


if __name__ == "__main__":
    unittest.main()
