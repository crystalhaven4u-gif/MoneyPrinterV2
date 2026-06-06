import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import farmer, storage
from longform.youtube_api import QuotaExceeded

NOW = datetime(2026, 1, 11, 0, 0, 0, tzinfo=timezone.utc)

WEIGHTS = {
    "trend_velocity": 0.30,
    "retention_proxy": 0.25,
    "search_demand": 0.15,
    "competition_gap": 0.15,
    "rpm_band": 0.15,
}
ARCHETYPES = {"story_narration": 0.85}
PROBE_BY_ID = {"n": {"id": "n", "default_format": "story_narration"}}  # rpm_band defaults to mid


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class HelperMathTests(unittest.TestCase):
    def test_parse_duration_seconds(self):
        self.assertEqual(farmer.parse_duration_seconds("PT45M"), 2700)
        self.assertEqual(farmer.parse_duration_seconds("PT1H2M3S"), 3723)
        self.assertEqual(farmer.parse_duration_seconds("PT31M"), 1860)
        self.assertEqual(farmer.parse_duration_seconds("P1DT2H"), 93600)
        self.assertEqual(farmer.parse_duration_seconds(""), 0)
        self.assertEqual(farmer.parse_duration_seconds("garbage"), 0)

    def test_passes_filters(self):
        self.assertTrue(farmer.passes_filters(1860, 2_000_000))
        self.assertTrue(farmer.passes_filters(1801, 1_000_001))
        self.assertFalse(farmer.passes_filters(1800, 2_000_000))   # duration not > 30min
        self.assertFalse(farmer.passes_filters(2000, 1_000_000))   # views not > 1M

    def test_view_velocity(self):
        velocity = farmer.compute_view_velocity(1000, "2026-01-01T00:00:00Z", NOW)
        self.assertAlmostEqual(velocity, 100.0)  # 1000 views / 10 days

    def test_view_velocity_floors_days_at_one(self):
        velocity = farmer.compute_view_velocity(500, "2026-01-11T00:00:00Z", NOW)
        self.assertAlmostEqual(velocity, 500.0)  # same-day -> divide by 1, not 0

    def test_outlier_score(self):
        self.assertAlmostEqual(farmer.compute_outlier_score(1000, 10), 100.0)
        self.assertAlmostEqual(farmer.compute_outlier_score(1000, 0), 1000.0)  # subs=0 guard

    def test_normalize(self):
        self.assertEqual(farmer._normalize([100, 300]), [0.0, 1.0])
        self.assertEqual(farmer._normalize([5, 5, 5]), [0.5, 0.5, 0.5])  # zero range
        self.assertEqual(farmer._normalize([]), [])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class ScoringTests(unittest.TestCase):
    def _survivors(self):
        return [
            {"video_id": "A", "niche": "n", "title": "A", "views": 1_000_000,
             "view_velocity": 100, "outlier_score": 10, "duration_s": 2700},
            {"video_id": "B", "niche": "n", "title": "B", "views": 3_000_000,
             "view_velocity": 300, "outlier_score": 40, "duration_s": 2700},
        ]

    def test_score_videos_is_deterministic(self):
        survivors = farmer.score_videos(self._survivors(), WEIGHTS, ARCHETYPES, PROBE_BY_ID)
        scores = {s["video_id"]: s["score"] for s in survivors}
        # A: 0.30*0 + 0.25*0.85 + 0.15*0 + 0.15*0 + 0.15*0.5
        self.assertEqual(scores["A"], 0.2875)
        # B: 0.30*1 + 0.25*0.85 + 0.15*1 + 0.15*1 + 0.15*0.5
        self.assertEqual(scores["B"], 0.8875)

    def test_min_score_gate_in_topics_slate(self):
        survivors = farmer.score_videos(self._survivors(), WEIGHTS, ARCHETYPES, PROBE_BY_ID)
        discovery = {
            "candidate_niche_probes": {"probes": [{"id": "n", "label": "N", "default_format": "story_narration"}]},
        }
        result = {
            "videos": survivors,
            "cc_by_niche": {"n": 42},
            "min_score_to_queue": 0.6,
            "weights": WEIGHTS,
            "stopped_early": False,
        }
        slate = farmer.build_topics_slate(result, discovery, now=NOW)

        self.assertEqual(len(slate["niches"]), 1)
        niche = slate["niches"][0]
        self.assertEqual(niche["video_count"], 1)  # only B clears 0.6
        self.assertEqual(niche["examples"][0]["video_id"], "B")
        self.assertEqual(niche["cc_feasibility"], 42)
        self.assertEqual(niche["aggregate_score"], 0.8875)


# --------------------------------------------------------------------------- #
# farm() integration with a fake client
# --------------------------------------------------------------------------- #
class FakeClient:
    quota_used = 7

    def search_list(self, **params):
        if params.get("videoLicense") == "creativeCommon":
            return {"pageInfo": {"totalResults": 123}, "items": []}
        return {
            "items": [
                {"id": {"videoId": "vid_pass"}},
                {"id": {"videoId": "vid_fail"}},
            ]
        }

    def videos_list(self, ids, part="contentDetails,statistics,snippet"):
        catalog = {
            "vid_pass": {
                "id": "vid_pass",
                "contentDetails": {"duration": "PT45M"},
                "statistics": {"viewCount": "2000000"},
                "snippet": {"title": "The Man Who Did It", "publishedAt": "2026-01-01T00:00:00Z", "channelId": "ch1"},
            },
            "vid_fail": {
                "id": "vid_fail",
                "contentDetails": {"duration": "PT10M"},   # too short
                "statistics": {"viewCount": "500"},         # too few views
                "snippet": {"title": "Short clip", "publishedAt": "2026-01-01T00:00:00Z", "channelId": "ch2"},
            },
        }
        return {"items": [catalog[i] for i in ids if i in catalog]}

    def channels_list(self, ids, part="statistics"):
        catalog = {"ch1": {"id": "ch1", "statistics": {"subscriberCount": "10000"}}}
        return {"items": [catalog[i] for i in ids if i in catalog]}


class QuotaStopClient(FakeClient):
    def search_list(self, **params):
        if params.get("q") == "q2":
            raise QuotaExceeded("budget hit")
        return super().search_list(**params)


def _discovery(probes):
    return {
        "scout_scoring": {"weights": WEIGHTS, "hard_gates": {"min_score_to_queue": 0.6}},
        "format_archetypes": ARCHETYPES,
        "candidate_niche_probes": {"probes": probes},
    }


class FarmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "farm.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_farm_filters_enriches_and_persists(self):
        discovery = _discovery([{"id": "n1", "label": "N1", "queries": ["q1"], "default_format": "story_narration"}])
        result = farmer.farm(FakeClient(), discovery, db_path=self.db_path, now=NOW)

        self.assertEqual(len(result["videos"]), 1)  # vid_fail filtered out
        survivor = result["videos"][0]
        self.assertEqual(survivor["video_id"], "vid_pass")
        self.assertEqual(survivor["views"], 2_000_000)
        self.assertEqual(survivor["duration_s"], 2700)
        self.assertEqual(survivor["subs"], 10000)
        self.assertAlmostEqual(survivor["outlier_score"], 200.0)
        self.assertEqual(survivor["cc_feasibility"], 123)
        self.assertIn("score", survivor)
        self.assertFalse(result["stopped_early"])
        self.assertEqual(result["cc_by_niche"]["n1"], 123)

        persisted = storage.get_all(db_path=self.db_path)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["video_id"], "vid_pass")

    def test_farm_is_idempotent(self):
        discovery = _discovery([{"id": "n1", "queries": ["q1"], "default_format": "story_narration"}])
        farmer.farm(FakeClient(), discovery, db_path=self.db_path, now=NOW)
        farmer.farm(FakeClient(), discovery, db_path=self.db_path, now=NOW)
        self.assertEqual(len(storage.get_all(db_path=self.db_path)), 1)  # upsert, no dupes

    def test_farm_stops_gracefully_on_quota(self):
        discovery = _discovery([
            {"id": "n1", "queries": ["q1"], "default_format": "story_narration"},
            {"id": "n2", "queries": ["q2"], "default_format": "story_narration"},
        ])
        result = farmer.farm(QuotaStopClient(), discovery, db_path=self.db_path, now=NOW)

        self.assertTrue(result["stopped_early"])
        # n1 still farmed and persisted; n2 aborted.
        self.assertEqual(result["cc_by_niche"], {"n1": 123})
        self.assertEqual(len(result["videos"]), 1)


if __name__ == "__main__":
    unittest.main()
