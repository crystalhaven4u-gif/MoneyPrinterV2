import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import base64

from longform import image_providers, storage
from longform.image_providers import ImageResult, aspect_to_dimensions, generate_image


# --------------------------------------------------------------------------- #
# Fakes -- nothing here touches the network.
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, content=b"", json_data=None, text=""):
        self.status_code = status_code
        self.content = content
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class FakeSession:
    """Routes .request() to a handler and records every call."""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._handler(method, url, kwargs)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


FREE_CFG = {
    "default_provider": "pollinations",
    "fallback_order": ["pollinations", "cloudflare", "gemini"],
    "cloudflare": {"account_id": "", "api_token": ""},
    "local_sd_url": "",
    "gemini_quality": "standard",
    "gemini": {"api_key": "", "base_url": "https://gemini.test", "model": "m"},
}


def _cfg(**overrides):
    cfg = {
        "default_provider": "pollinations",
        "fallback_order": ["pollinations", "cloudflare", "gemini"],
        "cloudflare": {"account_id": "", "api_token": ""},
        "local_sd_url": "",
        "gemini_quality": "standard",
        "gemini": {"api_key": "", "base_url": "https://gemini.test", "model": "m"},
    }
    cfg.update(overrides)
    return cfg


class AspectMappingTests(unittest.TestCase):
    def test_known_ratios(self):
        self.assertEqual(aspect_to_dimensions("16:9"), (1280, 720))
        self.assertEqual(aspect_to_dimensions("9:16"), (720, 1280))
        self.assertEqual(aspect_to_dimensions("1:1"), (1024, 1024))

    def test_unknown_landscape_ratio_scaled_to_long_edge(self):
        width, height = aspect_to_dimensions("5:2")
        self.assertEqual(width, 1280)
        self.assertLess(height, width)
        self.assertEqual(height % 8, 0)

    def test_unknown_portrait_ratio_scaled_to_long_edge(self):
        width, height = aspect_to_dimensions("2:3")
        self.assertEqual(height, 1280)
        self.assertLess(width, height)
        self.assertEqual(width % 8, 0)

    def test_wxh_separator_supported(self):
        self.assertEqual(aspect_to_dimensions("16x9"), (1280, 720))

    def test_garbage_falls_back_to_default(self):
        self.assertEqual(aspect_to_dimensions("not-a-ratio"), (1280, 720))
        self.assertEqual(aspect_to_dimensions(""), (1280, 720))


class FallbackChainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._tmp.name, "img.png")

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_provider_used_when_it_succeeds(self):
        def handler(method, url, kwargs):
            self.assertIn("pollinations.ai", url)
            return FakeResponse(200, content=b"PNGBYTES")

        session = FakeSession(handler)
        result = generate_image(
            "a cat", output_path=self.out, config=FREE_CFG, session=session, log=False
        )

        self.assertEqual(result.provider, "pollinations")
        self.assertFalse(result.is_placeholder)
        self.assertEqual(result.cost, 0.0)
        self.assertEqual(len(session.calls), 1)  # no fallback needed
        with open(self.out, "rb") as handle:
            self.assertEqual(handle.read(), b"PNGBYTES")

    def test_falls_back_to_next_eligible_provider_on_failure(self):
        cfg = _cfg(cloudflare={"account_id": "acct", "api_token": "tok"})

        def handler(method, url, kwargs):
            if "pollinations.ai" in url:
                return FakeResponse(503, text="upstream down")  # fails -> fall back
            if "cloudflare.com" in url:
                return FakeResponse(200, json_data={"result": {"image": _b64(b"CF")}})
            raise AssertionError(f"unexpected url {url}")

        session = FakeSession(handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session, log=False
        )

        self.assertEqual(result.provider, "cloudflare")
        self.assertFalse(result.is_placeholder)
        with open(self.out, "rb") as handle:
            self.assertEqual(handle.read(), b"CF")

    def test_unconfigured_providers_are_skipped_without_calling(self):
        # cloudflare not configured + gemini not high-quality -> both skipped;
        # pollinations is the only eligible provider.
        def handler(method, url, kwargs):
            if "pollinations.ai" in url:
                return FakeResponse(200, content=b"OK")
            raise AssertionError(f"should not call {url}")

        session = FakeSession(handler)
        result = generate_image(
            "x", output_path=self.out, config=FREE_CFG, session=session, log=False
        )
        self.assertEqual(result.provider, "pollinations")
        self.assertEqual(len(session.calls), 1)

    def test_all_fail_writes_flagged_placeholder_png(self):
        import requests as real_requests

        def handler(method, url, kwargs):
            raise real_requests.RequestException("network down")

        session = FakeSession(handler)
        result = generate_image(
            "x",
            aspect_ratio="16:9",
            output_path=self.out,
            config=FREE_CFG,
            session=session,
            log=False,
        )

        self.assertTrue(result.is_placeholder)
        self.assertEqual(result.provider, "placeholder")
        with open(self.out, "rb") as handle:
            header = handle.read(8)
        self.assertEqual(header, b"\x89PNG\r\n\x1a\n")  # valid PNG signature

    def test_gemini_only_eligible_at_high_quality(self):
        cfg = _cfg(
            default_provider="gemini",
            fallback_order=["gemini", "pollinations"],
            gemini={"api_key": "k", "base_url": "https://gemini.test", "model": "m"},
        )

        # standard quality: gemini skipped, falls through to pollinations.
        def standard_handler(method, url, kwargs):
            if "gemini.test" in url:
                raise AssertionError("gemini must not be called at standard quality")
            return FakeResponse(200, content=b"POLL")

        session = FakeSession(standard_handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session,
            quality="standard", log=False,
        )
        self.assertEqual(result.provider, "pollinations")

        # high quality + key: gemini is attempted first and used.
        def high_handler(method, url, kwargs):
            self.assertIn("gemini.test", url)
            return FakeResponse(
                200,
                json_data={
                    "candidates": [
                        {"content": {"parts": [
                            {"inlineData": {"mimeType": "image/png", "data": _b64(b"GEM")}}
                        ]}}
                    ]
                },
            )

        session = FakeSession(high_handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session,
            quality="high", log=False,
        )
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.cost, image_providers.PROVIDER_COST["gemini"])
        with open(self.out, "rb") as handle:
            self.assertEqual(handle.read(), b"GEM")


class BackendDecodeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._tmp.name, "img.png")

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_sd_decodes_first_image(self):
        cfg = _cfg(default_provider="local_sd", local_sd_url="http://127.0.0.1:7860")

        def handler(method, url, kwargs):
            self.assertIn("/sdapi/v1/txt2img", url)
            return FakeResponse(200, json_data={"images": [_b64(b"SD")]})

        session = FakeSession(handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session, log=False
        )
        self.assertEqual(result.provider, "local_sd")
        with open(self.out, "rb") as handle:
            self.assertEqual(handle.read(), b"SD")

    def test_pollinations_sends_referrer_and_token_when_configured(self):
        cfg = _cfg(pollinations={"referrer": "mpv2-longform", "token": "secret"})
        captured = {}

        def handler(method, url, kwargs):
            captured["params"] = kwargs.get("params", {})
            captured["headers"] = kwargs.get("headers", {})
            return FakeResponse(200, content=b"OK")

        session = FakeSession(handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session, log=False
        )
        self.assertEqual(result.provider, "pollinations")
        self.assertEqual(captured["params"].get("referrer"), "mpv2-longform")
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer secret")

    def test_pollinations_omits_referrer_token_when_unset(self):
        captured = {}

        def handler(method, url, kwargs):
            captured["params"] = kwargs.get("params", {})
            captured["headers"] = kwargs.get("headers", {})
            return FakeResponse(200, content=b"OK")

        session = FakeSession(handler)
        generate_image(
            "x", output_path=self.out, config=FREE_CFG, session=session, log=False
        )
        self.assertNotIn("referrer", captured["params"])
        self.assertNotIn("Authorization", captured["headers"])

    def test_provider_arg_overrides_default(self):
        # thumbnail_provider-style override: force cloudflare first.
        cfg = _cfg(cloudflare={"account_id": "a", "api_token": "t"})

        def handler(method, url, kwargs):
            self.assertIn("cloudflare.com", url)
            return FakeResponse(200, json_data={"result": {"image": _b64(b"CF")}})

        session = FakeSession(handler)
        result = generate_image(
            "x", output_path=self.out, config=cfg, session=session,
            provider="cloudflare", log=False,
        )
        self.assertEqual(result.provider, "cloudflare")


class NoNetworkAndLoggingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._tmp.name, "img.png")
        self.db = os.path.join(self._tmp.name, "farm.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_real_requests_never_used_when_session_injected(self):
        # If anything reaches module-level requests, fail loudly.
        with patch.object(image_providers, "requests") as fake_requests:
            fake_requests.request.side_effect = AssertionError("real network used!")

            def handler(method, url, kwargs):
                return FakeResponse(200, content=b"OK")

            session = FakeSession(handler)
            result = generate_image(
                "x", output_path=self.out, config=FREE_CFG, session=session, log=False
            )
            self.assertEqual(result.provider, "pollinations")
            fake_requests.request.assert_not_called()

    def test_each_image_logged_with_provider_and_cost(self):
        def handler(method, url, kwargs):
            return FakeResponse(200, content=b"OK")

        session = FakeSession(handler)
        generate_image(
            "a sleepy fox",
            aspect_ratio="16:9",
            output_path=self.out,
            config=FREE_CFG,
            session=session,
            db_path=self.db,
            log=True,
        )

        rows = storage.get_image_log(db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "pollinations")
        self.assertEqual(rows[0]["cost"], 0.0)
        self.assertEqual(rows[0]["is_placeholder"], 0)
        self.assertEqual(rows[0]["aspect_ratio"], "16:9")
        self.assertEqual(rows[0]["width"], 1280)

    def test_placeholder_is_logged_and_flagged(self):
        import requests as real_requests

        def handler(method, url, kwargs):
            raise real_requests.RequestException("down")

        session = FakeSession(handler)
        generate_image(
            "x", output_path=self.out, config=FREE_CFG, session=session,
            db_path=self.db, log=True,
        )

        rows = storage.get_image_log(db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "placeholder")
        self.assertEqual(rows[0]["is_placeholder"], 1)
        self.assertTrue(rows[0]["note"])  # captured why each provider failed


if __name__ == "__main__":
    unittest.main()
