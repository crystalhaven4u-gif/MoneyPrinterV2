import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import ledger


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._temp_dir.name, "ledger.db")

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_build_utm_campaign(self) -> None:
        self.assertEqual(
            ledger.build_utm_campaign("offszn", "abc123"),
            "mpv2-offszn-abc123",
        )

    def test_create_entry_sets_defaults_and_utm(self) -> None:
        video_id = ledger.create_entry(
            lane="offszn",
            topic="earbuds",
            prompt_version="v1",
            hook_id="h1",
            voice="Jasper",
            db_path=self.db_path,
        )

        entry = ledger.get_entry(video_id, db_path=self.db_path)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["lane"], "offszn")
        self.assertEqual(entry["topic"], "earbuds")
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["voice"], "Jasper")
        self.assertEqual(entry["utm_campaign"], f"mpv2-offszn-{video_id}")
        self.assertTrue(entry["created_at"])

    def test_create_entry_honours_explicit_id(self) -> None:
        video_id = ledger.create_entry(
            lane="faceless", video_id="fixed-id", db_path=self.db_path
        )
        self.assertEqual(video_id, "fixed-id")
        entry = ledger.get_entry("fixed-id", db_path=self.db_path)
        self.assertEqual(entry["utm_campaign"], "mpv2-faceless-fixed-id")

    def test_create_entry_rejects_invalid_status(self) -> None:
        with self.assertRaises(ValueError):
            ledger.create_entry(lane="offszn", status="bogus", db_path=self.db_path)

    def test_update_entry_updates_fields(self) -> None:
        video_id = ledger.create_entry(lane="offszn", db_path=self.db_path)
        ledger.update_entry(
            video_id, db_path=self.db_path, topic="new topic", video_path="/x.mp4"
        )
        entry = ledger.get_entry(video_id, db_path=self.db_path)
        self.assertEqual(entry["topic"], "new topic")
        self.assertEqual(entry["video_path"], "/x.mp4")

    def test_update_entry_rejects_unknown_column(self) -> None:
        video_id = ledger.create_entry(lane="offszn", db_path=self.db_path)
        with self.assertRaises(ValueError):
            ledger.update_entry(video_id, db_path=self.db_path, bogus="x")

    def test_update_entry_rejects_invalid_status(self) -> None:
        video_id = ledger.create_entry(lane="offszn", db_path=self.db_path)
        with self.assertRaises(ValueError):
            ledger.update_entry(video_id, db_path=self.db_path, status="nope")

    def test_update_missing_entry_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            ledger.update_entry("missing", db_path=self.db_path, topic="x")

    def test_set_status(self) -> None:
        video_id = ledger.create_entry(lane="offszn", db_path=self.db_path)
        ledger.set_status(video_id, "approved", db_path=self.db_path)
        self.assertEqual(
            ledger.get_entry(video_id, db_path=self.db_path)["status"], "approved"
        )

    def test_mark_published(self) -> None:
        video_id = ledger.create_entry(lane="offszn", db_path=self.db_path)
        ledger.mark_published(
            video_id, "youtube", "yt-123", db_path=self.db_path
        )
        entry = ledger.get_entry(video_id, db_path=self.db_path)
        self.assertEqual(entry["status"], "published")
        self.assertEqual(entry["platform"], "youtube")
        self.assertEqual(entry["platform_post_id"], "yt-123")

    def test_list_entries_filters_by_status(self) -> None:
        a = ledger.create_entry(lane="offszn", db_path=self.db_path)
        b = ledger.create_entry(lane="faceless", db_path=self.db_path)
        ledger.set_status(b, "approved", db_path=self.db_path)

        pending = ledger.list_entries(status="pending", db_path=self.db_path)
        approved = ledger.list_entries(status="approved", db_path=self.db_path)
        all_entries = ledger.list_entries(db_path=self.db_path)

        self.assertEqual([e["id"] for e in pending], [a])
        self.assertEqual([e["id"] for e in approved], [b])
        self.assertEqual(len(all_entries), 2)

    def test_get_missing_entry_returns_none(self) -> None:
        self.assertIsNone(ledger.get_entry("nope", db_path=self.db_path))


if __name__ == "__main__":
    unittest.main()
