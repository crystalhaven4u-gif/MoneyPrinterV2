import os
import sys
import unittest
from unittest.mock import Mock, patch

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from longform import youtube_api
from longform.youtube_api import YouTubeDataClient, YouTubeAPIError, QuotaExceeded


class MockResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json_data


class YouTubeDataClientTests(unittest.TestCase):
    def setUp(self):
        # Never actually sleep during retry/backoff in tests.
        self._sleep_patch = patch.object(youtube_api, "_sleep", lambda *_: None)
        self._sleep_patch.start()

    def tearDown(self):
        self._sleep_patch.stop()

    def test_quota_cost_tracked_per_call(self):
        session = Mock()
        session.request.side_effect = [
            MockResponse(200, {"items": []}),  # search.list = 100
            MockResponse(200, {"items": []}),  # videos.list = 1
        ]
        client = YouTubeDataClient("k", daily_budget=9000, session=session)

        client.search_list(q="x", type="video")
        client.videos_list(["a", "b"])

        self.assertEqual(client.quota_used, 101)
        self.assertEqual(client.quota_remaining, 9000 - 101)

    def test_budget_stop_before_spending(self):
        session = Mock()
        # Only ONE real call should ever happen.
        session.request.side_effect = [MockResponse(200, {"items": []})]
        client = YouTubeDataClient("k", daily_budget=100, session=session)

        client.search_list(q="x", type="video")  # spends exactly 100
        self.assertEqual(client.quota_used, 100)

        # The next search.list would exceed 100 -> raise before calling the API.
        with self.assertRaises(QuotaExceeded):
            client.search_list(q="y", type="video")

        self.assertEqual(session.request.call_count, 1)
        self.assertEqual(client.quota_used, 100)

    def test_can_afford(self):
        client = YouTubeDataClient("k", daily_budget=150)
        self.assertTrue(client.can_afford("search.list"))
        client.quota_used = 60
        self.assertFalse(client.can_afford("search.list"))  # 60+100 > 150
        self.assertTrue(client.can_afford("videos.list"))   # 60+1 <= 150

    def test_retry_then_success_charges_quota_once(self):
        session = Mock()
        session.request.side_effect = [
            MockResponse(500, {}),
            MockResponse(200, {"items": [{"id": "v"}]}),
        ]
        client = YouTubeDataClient("k", daily_budget=9000, session=session)

        result = client.videos_list(["v"])

        self.assertEqual(result, {"items": [{"id": "v"}]})
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(client.quota_used, 1)  # charged once, only on success

    def test_network_exception_is_retried(self):
        session = Mock()
        session.request.side_effect = [
            requests.RequestException("boom"),
            MockResponse(200, {"items": []}),
        ]
        client = YouTubeDataClient("k", session=session)

        result = client.search_list(q="x")
        self.assertEqual(result, {"items": []})
        self.assertEqual(session.request.call_count, 2)

    def test_persistent_5xx_raises_and_does_not_charge(self):
        session = Mock()
        session.request.side_effect = [MockResponse(503, {}) for _ in range(4)]
        client = YouTubeDataClient("k", session=session, max_retries=4)

        with self.assertRaises(YouTubeAPIError):
            client.videos_list(["v"])
        self.assertEqual(client.quota_used, 0)

    def test_non_retryable_4xx_raises_immediately(self):
        session = Mock()
        session.request.side_effect = [
            MockResponse(403, {"error": {"message": "quotaExceeded"}})
        ]
        client = YouTubeDataClient("k", session=session)

        with self.assertRaises(YouTubeAPIError):
            client.search_list(q="x")
        self.assertEqual(session.request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
