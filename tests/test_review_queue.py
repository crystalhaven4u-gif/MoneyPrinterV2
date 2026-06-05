import json
import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import ledger
import review_queue


class ReviewQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = os.path.join(self._temp_dir.name, "review_queue")
        self.db_path = os.path.join(self._temp_dir.name, "ledger.db")
        self.source_dir = os.path.join(self._temp_dir.name, "src")
        os.makedirs(self.source_dir, exist_ok=True)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _make_video(self, name: str = "video.mp4") -> str:
        path = os.path.join(self.source_dir, name)
        with open(path, "wb") as handle:
            handle.write(b"\x00\x01\x02")
        return path

    def _metadata(self) -> dict:
        return {
            "lane": "offszn",
            "title": "OffSzn buds",
            "description": "AI generated",
            "hashtags": ["#offszn"],
            "target_platforms": ["youtube", "tiktok"],
            "hook_id": "h1",
            "prompt_version": "v1",
            "utm_campaign": "mpv2-offszn-vid1",
        }

    def _seed_ledger(self, video_id: str) -> None:
        ledger.create_entry(
            lane="offszn", video_id=video_id, db_path=self.db_path
        )

    def test_submit_creates_pending_files(self) -> None:
        video_path = self._make_video()
        record = review_queue.submit(
            video_path=video_path,
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
            update_ledger=False,
        )

        self.assertEqual(record["status"], "pending")
        self.assertEqual(record["video_id"], "vid1")
        self.assertTrue(
            os.path.exists(os.path.join(self.base_dir, "pending", "vid1.mp4"))
        )
        meta_path = os.path.join(self.base_dir, "pending", "vid1.json")
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["title"], "OffSzn buds")

    def test_submit_missing_required_field_raises(self) -> None:
        video_path = self._make_video()
        metadata = self._metadata()
        del metadata["hashtags"]
        with self.assertRaises(ValueError):
            review_queue.submit(
                video_path=video_path,
                metadata=metadata,
                video_id="vid1",
                base_dir=self.base_dir,
                update_ledger=False,
            )

    def test_submit_missing_video_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            review_queue.submit(
                video_path=os.path.join(self.source_dir, "nope.mp4"),
                metadata=self._metadata(),
                video_id="vid1",
                base_dir=self.base_dir,
                update_ledger=False,
            )

    def test_submit_move_removes_source(self) -> None:
        video_path = self._make_video()
        review_queue.submit(
            video_path=video_path,
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            move=True,
            update_ledger=False,
        )
        self.assertFalse(os.path.exists(video_path))
        self.assertTrue(
            os.path.exists(os.path.join(self.base_dir, "pending", "vid1.mp4"))
        )

    def test_submit_updates_ledger(self) -> None:
        self._seed_ledger("vid1")
        video_path = self._make_video()
        review_queue.submit(
            video_path=video_path,
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )
        entry = ledger.get_entry("vid1", db_path=self.db_path)
        self.assertEqual(entry["status"], "pending")
        self.assertTrue(entry["video_path"].endswith(os.path.join("pending", "vid1.mp4")))

    def test_list_pending(self) -> None:
        review_queue.submit(
            video_path=self._make_video("a.mp4"),
            metadata=self._metadata(),
            video_id="a",
            base_dir=self.base_dir,
            update_ledger=False,
        )
        review_queue.submit(
            video_path=self._make_video("b.mp4"),
            metadata=self._metadata(),
            video_id="b",
            base_dir=self.base_dir,
            update_ledger=False,
        )
        ids = [item["video_id"] for item in review_queue.list_pending(self.base_dir)]
        self.assertEqual(sorted(ids), ["a", "b"])

    def test_approve_moves_files_and_updates_ledger(self) -> None:
        self._seed_ledger("vid1")
        review_queue.submit(
            video_path=self._make_video(),
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )

        item = review_queue.approve("vid1", base_dir=self.base_dir, db_path=self.db_path)

        self.assertEqual(item["state"], "approved")
        self.assertFalse(
            os.path.exists(os.path.join(self.base_dir, "pending", "vid1.mp4"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_dir, "approved", "vid1.mp4"))
        )
        self.assertEqual(
            ledger.get_entry("vid1", db_path=self.db_path)["status"], "approved"
        )

    def test_reject_moves_to_rejected(self) -> None:
        self._seed_ledger("vid1")
        review_queue.submit(
            video_path=self._make_video(),
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )

        review_queue.reject("vid1", base_dir=self.base_dir, db_path=self.db_path)

        self.assertTrue(
            os.path.exists(os.path.join(self.base_dir, "rejected", "vid1.json"))
        )
        self.assertEqual(
            ledger.get_entry("vid1", db_path=self.db_path)["status"], "rejected"
        )

    def test_approve_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            review_queue.approve("nope", base_dir=self.base_dir, db_path=self.db_path)

    def test_mark_published_moves_and_updates_ledger(self) -> None:
        self._seed_ledger("vid1")
        review_queue.submit(
            video_path=self._make_video(),
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )
        review_queue.approve("vid1", base_dir=self.base_dir, db_path=self.db_path)

        review_queue.mark_published(
            "vid1",
            platform="youtube",
            platform_post_id="yt-1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )

        self.assertTrue(
            os.path.exists(os.path.join(self.base_dir, "published", "vid1.mp4"))
        )
        entry = ledger.get_entry("vid1", db_path=self.db_path)
        self.assertEqual(entry["status"], "published")
        self.assertEqual(entry["platform"], "youtube")
        self.assertEqual(entry["platform_post_id"], "yt-1")

    def test_approve_without_ledger_entry_still_moves(self) -> None:
        # No ledger row seeded: filesystem bookkeeping must still succeed.
        review_queue.submit(
            video_path=self._make_video(),
            metadata=self._metadata(),
            video_id="vid1",
            base_dir=self.base_dir,
            db_path=self.db_path,
        )
        item = review_queue.approve("vid1", base_dir=self.base_dir, db_path=self.db_path)
        self.assertEqual(item["state"], "approved")


if __name__ == "__main__":
    unittest.main()
