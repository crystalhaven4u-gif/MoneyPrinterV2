import os
import sys
import types
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

fake_kittentts = types.ModuleType("kittentts")
fake_kittentts.KittenTTS = object
sys.modules.setdefault("kittentts", fake_kittentts)

fake_ollama = types.ModuleType("ollama")
fake_ollama.Client = object
sys.modules.setdefault("ollama", fake_ollama)

fake_llm_provider = types.ModuleType("llm_provider")
fake_llm_provider.select_model = lambda model: None
sys.modules.setdefault("llm_provider", fake_llm_provider)

fake_tts_module = types.ModuleType("classes.Tts")
fake_tts_module.TTS = object
sys.modules.setdefault("classes.Tts", fake_tts_module)

fake_twitter_module = types.ModuleType("classes.Twitter")
fake_twitter_module.Twitter = object
sys.modules.setdefault("classes.Twitter", fake_twitter_module)

fake_youtube_module = types.ModuleType("classes.YouTube")
fake_youtube_module.YouTube = object
sys.modules.setdefault("classes.YouTube", fake_youtube_module)

import cron


class CronReviewQueueTests(unittest.TestCase):
    @patch("cron.review_queue")
    @patch("cron.YouTube")
    @patch("cron.TTS")
    @patch("cron.get_accounts")
    @patch("cron.select_model")
    @patch("cron.get_verbose")
    def test_youtube_cron_submits_to_review_queue_without_uploading(
        self,
        get_verbose_mock,
        select_model_mock,
        get_accounts_mock,
        tts_cls_mock,
        youtube_cls_mock,
        review_queue_mock,
    ) -> None:
        get_verbose_mock.return_value = False
        get_accounts_mock.return_value = [
            {
                "id": "yt-1",
                "nickname": "Channel",
                "firefox_profile": "/tmp/profile",
                "niche": "finance",
                "language": "English",
            }
        ]
        youtube_instance = youtube_cls_mock.return_value
        youtube_instance.video_path = "/tmp/video.mp4"
        youtube_instance.video_id = "vid-1"
        youtube_instance.build_review_metadata.return_value = {"title": "Title"}

        review_queue_mock.default_target_platforms.return_value = ["youtube"]
        review_queue_mock.submit.return_value = {"video_id": "vid-1"}

        with patch.object(
            sys,
            "argv",
            ["cron.py", "youtube", "yt-1", "llama3.2:3b"],
        ):
            cron.main()

        select_model_mock.assert_called_once_with("llama3.2:3b")
        tts_cls_mock.assert_called_once()
        youtube_instance.generate_video.assert_called_once()
        # Distribution safety: cron must never publish directly.
        youtube_instance.upload_video.assert_not_called()
        review_queue_mock.submit.assert_called_once()
        submit_kwargs = review_queue_mock.submit.call_args.kwargs
        self.assertEqual(submit_kwargs["video_id"], "vid-1")
        self.assertEqual(submit_kwargs["video_path"], "/tmp/video.mp4")


if __name__ == "__main__":
    unittest.main()
