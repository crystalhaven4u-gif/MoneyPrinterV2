"""
Pluggable image-generation backends for the long-form engine.

The engine must never hard-depend on a single paid backend, so this module
exposes ONE entry point -- ``generate_image()`` -- that tries a chosen provider
and, on failure, falls back down a configurable chain. The default chain is free
and needs no API key:

    pollinations  (free, keyless GET)         -> default
    cloudflare    (Workers AI Flux, free tier)
    gemini        (paid; only when quality="high" AND a key is configured)
    local_sd      (optional; a local SD HTTP API if a URL is configured)

If every eligible provider fails, a plain colored placeholder PNG is written and
flagged (``is_placeholder=True``) so an image step never crashes a render.

Each call is logged to the long-form ledger (``.mp/farm.db`` ``image_log``
table) with the provider and per-image cost (0 for the free backends).

Self-contained: depends only on ``requests`` + stdlib (no Pillow), consistent
with the rest of ``src/longform/``. The placeholder PNG is encoded by hand with
``zlib``/``struct`` to avoid an imaging dependency.
"""

import base64
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import requests

# Repo root = three levels up (src/longform/image_providers.py).
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Aspect ratio -> (width, height). 16:9 long-form is the common case.
ASPECT_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1280, 544),
    "4:5": (864, 1080),
}
DEFAULT_DIMENSIONS = (1280, 720)
LONG_EDGE = 1280  # used when deriving dimensions for an unmapped ratio

# Per-image cost in USD. Free backends are exactly 0; Gemini is a rough estimate
# (image preview pricing) used only for ledger accounting, not billing.
PROVIDER_COST = {
    "pollinations": 0.0,
    "cloudflare": 0.0,
    "local_sd": 0.0,
    "gemini": 0.03,
    "placeholder": 0.0,
}

DEFAULT_FALLBACK_ORDER = ["pollinations", "cloudflare", "gemini"]

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-3.1-flash-image-preview"

# Neutral slate-grey placeholder so a failed image is obvious but not jarring.
PLACEHOLDER_RGB = (40, 44, 52)


class ImageProviderError(RuntimeError):
    """Raised by a backend when it cannot produce an image."""


@dataclass
class ImageResult:
    """The outcome of a generate_image() call."""

    path: str
    provider: str
    cost: float
    is_placeholder: bool
    width: int
    height: int


# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #
def _round8(value: float) -> int:
    """Rounds to the nearest multiple of 8 (diffusion models like /8 dims)."""
    return max(8, int(round(value / 8.0)) * 8)


def aspect_to_dimensions(aspect_ratio: str) -> tuple:
    """
    Maps an aspect ratio to (width, height).

    Known ratios use a curated size; an unmapped but parseable ``"w:h"`` (or
    ``"wxh"``) is scaled to a 1280px long edge; anything else falls back to
    16:9.
    """
    if not aspect_ratio:
        return DEFAULT_DIMENSIONS
    key = str(aspect_ratio).strip()
    if key in ASPECT_DIMENSIONS:
        return ASPECT_DIMENSIONS[key]

    lowered = key.lower()
    separator = ":" if ":" in lowered else ("x" if "x" in lowered else None)
    if separator:
        try:
            left, right = lowered.split(separator)
            width_ratio, height_ratio = float(left), float(right)
            if width_ratio > 0 and height_ratio > 0:
                if width_ratio >= height_ratio:
                    return (LONG_EDGE, _round8(LONG_EDGE * height_ratio / width_ratio))
                return (_round8(LONG_EDGE * width_ratio / height_ratio), LONG_EDGE)
        except (ValueError, ZeroDivisionError):
            pass
    return DEFAULT_DIMENSIONS


# --------------------------------------------------------------------------- #
# Backends -- each returns raw image bytes or raises.
# --------------------------------------------------------------------------- #
def _raise_for_status(response) -> None:
    code = getattr(response, "status_code", 200)
    if code != 200:
        detail = ""
        try:
            detail = (response.text or "")[:200]
        except Exception:  # pragma: no cover - defensive
            detail = ""
        raise ImageProviderError(f"HTTP {code}: {detail}".strip())


def _provider_pollinations(prompt, width, height, aspect_ratio, cfg, session) -> bytes:
    """Free, keyless Flux endpoint. Returns the raw image bytes directly."""
    url = POLLINATIONS_URL.format(prompt=quote(prompt, safe=""))
    params = {"width": width, "height": height, "nologo": "true", "model": "flux"}
    response = session.request("GET", url, params=params, timeout=120)
    _raise_for_status(response)
    content = getattr(response, "content", b"")
    if not content:
        raise ImageProviderError("pollinations returned an empty body")
    return content


def _provider_cloudflare(prompt, width, height, aspect_ratio, cfg, session) -> bytes:
    """Cloudflare Workers AI Flux (free tier). Returns base64 in result.image."""
    cloudflare = cfg.get("cloudflare") or {}
    account_id = cloudflare.get("account_id")
    api_token = cloudflare.get("api_token")
    if not account_id or not api_token:
        raise ImageProviderError("cloudflare account_id/api_token not configured")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CLOUDFLARE_MODEL}"
    )
    response = session.request(
        "POST",
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        json={"prompt": prompt},
        timeout=120,
    )
    _raise_for_status(response)
    payload = response.json()
    encoded = (payload.get("result") or {}).get("image")
    if not encoded:
        raise ImageProviderError("cloudflare returned no image")
    return base64.b64decode(encoded)


def _provider_local_sd(prompt, width, height, aspect_ratio, cfg, session) -> bytes:
    """Local Stable Diffusion HTTP API (Automatic1111-style txt2img)."""
    base_url = (cfg.get("local_sd_url") or "").rstrip("/")
    if not base_url:
        raise ImageProviderError("local_sd_url not configured")
    url = f"{base_url}/sdapi/v1/txt2img"
    response = session.request(
        "POST",
        url,
        json={"prompt": prompt, "width": width, "height": height, "steps": 20},
        timeout=300,
    )
    _raise_for_status(response)
    payload = response.json()
    images = payload.get("images") or []
    if not images:
        raise ImageProviderError("local SD returned no images")
    return base64.b64decode(images[0])


def _provider_gemini(prompt, width, height, aspect_ratio, cfg, session) -> bytes:
    """Paid Gemini/Imagen image path; mirrors the existing Nano Banana 2 call."""
    gemini = cfg.get("gemini") or {}
    api_key = gemini.get("api_key")
    if not api_key:
        raise ImageProviderError("gemini api key not configured")
    base_url = (gemini.get("base_url") or GEMINI_BASE).rstrip("/")
    model = gemini.get("model") or GEMINI_MODEL
    url = f"{base_url}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio},
        },
    }
    response = session.request(
        "POST",
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    _raise_for_status(response)
    body = response.json()
    for candidate in body.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline or not inline.get("data"):
                continue
            mime = inline.get("mimeType") or inline.get("mime_type", "")
            if not mime or str(mime).startswith("image/"):
                return base64.b64decode(inline["data"])
    raise ImageProviderError("gemini returned no image payload")


_BACKENDS = {
    "pollinations": _provider_pollinations,
    "cloudflare": _provider_cloudflare,
    "local_sd": _provider_local_sd,
    "gemini": _provider_gemini,
}


def _provider_available(name: str, cfg: dict, quality: str) -> bool:
    """Whether a provider has what it needs to even be attempted."""
    if name == "pollinations":
        return True
    if name == "cloudflare":
        cloudflare = cfg.get("cloudflare") or {}
        return bool(cloudflare.get("account_id") and cloudflare.get("api_token"))
    if name == "local_sd":
        return bool(cfg.get("local_sd_url"))
    if name == "gemini":
        # Paid path: only when high quality is explicitly requested AND keyed.
        return quality == "high" and bool((cfg.get("gemini") or {}).get("api_key"))
    return False


# --------------------------------------------------------------------------- #
# Placeholder PNG (hand-encoded, stdlib only)
# --------------------------------------------------------------------------- #
def _encode_solid_png(width: int, height: int, rgb: tuple) -> bytes:
    """Encodes a solid-color RGB PNG without any imaging library."""
    import struct
    import zlib

    red, green, blue = rgb
    row = bytes([0]) + bytes([red, green, blue]) * width  # filter byte 0 + pixels
    compressed = zlib.compress(row * height, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


# --------------------------------------------------------------------------- #
# Output + config helpers
# --------------------------------------------------------------------------- #
def _write_bytes(path: str, data: bytes) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def _default_output_dir() -> str:
    directory = os.path.join(_ROOT_DIR, ".mp")
    os.makedirs(directory, exist_ok=True)
    return directory


def load_image_config() -> dict:
    """
    Loads the resolved image config from config.py, falling back to safe free
    defaults if config is unavailable (keeps the module importable standalone).
    """
    defaults = {
        "default_provider": "pollinations",
        "thumbnail_provider": "pollinations",
        "fallback_order": list(DEFAULT_FALLBACK_ORDER),
        "cloudflare": {"account_id": "", "api_token": ""},
        "local_sd_url": "",
        "gemini_quality": "standard",
        "gemini": {"api_key": "", "base_url": GEMINI_BASE, "model": GEMINI_MODEL},
    }
    try:
        import sys

        src_dir = os.path.join(_ROOT_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from config import get_image_config

        return get_image_config()
    except Exception:
        return defaults


def _build_chain(chosen: str, fallback_order) -> list:
    """The chosen provider first, then the fallback order, de-duplicated."""
    ordered = []
    seen = set()
    for name in [chosen, *(fallback_order or [])]:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _log_image(result: ImageResult, prompt, aspect_ratio, quality, db_path, note=""):
    """Best-effort ledger write; logging must never crash a render."""
    try:
        from . import storage

        storage.log_image(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "provider": result.provider,
                "prompt": (prompt or "")[:500],
                "aspect_ratio": aspect_ratio,
                "width": result.width,
                "height": result.height,
                "output_path": result.path,
                "cost": result.cost,
                "is_placeholder": int(result.is_placeholder),
                "quality": quality,
                "note": note,
            },
            db_path=db_path,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_image(
    prompt: str,
    aspect_ratio: str = "16:9",
    output_path: Optional[str] = None,
    quality: str = "standard",
    provider: Optional[str] = None,
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
    log: bool = True,
) -> ImageResult:
    """
    Generates an image, trying a provider chain and never crashing a render.

    Args:
        prompt (str): The image prompt.
        aspect_ratio (str): e.g. "16:9", "9:16", "1:1".
        output_path (str | None): Where to save the PNG. Defaults to a uuid file
            under ``.mp/``.
        quality (str): "standard" (free backends) or "high" (allows the paid
            Gemini backend when a key is configured).
        provider (str | None): Force the first provider tried; defaults to the
            configured ``default_provider``. (Packaging passes
            ``thumbnail_provider`` here for thumbnails.)
        config (dict | None): Resolved image config; loaded from config.py when
            omitted.
        session: A ``requests``-like object exposing ``.request(method, url,
            ...)``; defaults to ``requests``. Injected in tests so no real
            network call happens.
        db_path (str | None): Override the ledger DB path.
        log (bool): Whether to record the call in the ledger.

    Returns:
        ImageResult: saved ``path``, the ``provider`` used (or "placeholder"),
        the ``cost``, and ``is_placeholder``.
    """
    cfg = config if config is not None else load_image_config()
    http = session if session is not None else requests
    width, height = aspect_to_dimensions(aspect_ratio)
    chosen = provider or cfg.get("default_provider", "pollinations")
    chain = _build_chain(chosen, cfg.get("fallback_order", DEFAULT_FALLBACK_ORDER))

    if output_path is None:
        from uuid import uuid4

        output_path = os.path.join(_default_output_dir(), f"{uuid4().hex}.png")

    errors = []
    for name in chain:
        backend = _BACKENDS.get(name)
        if backend is None:
            errors.append(f"{name}: unknown provider")
            continue
        if not _provider_available(name, cfg, quality):
            errors.append(f"{name}: not eligible/configured")
            continue
        try:
            image_bytes = backend(prompt, width, height, aspect_ratio, cfg, http)
            if not image_bytes:
                raise ImageProviderError("empty image payload")
            _write_bytes(output_path, image_bytes)
            result = ImageResult(
                path=output_path,
                provider=name,
                cost=PROVIDER_COST.get(name, 0.0),
                is_placeholder=False,
                width=width,
                height=height,
            )
            if log:
                _log_image(result, prompt, aspect_ratio, quality, db_path)
            return result
        except Exception as exc:  # any failure -> try the next provider
            errors.append(f"{name}: {exc}")
            continue

    # Everything failed: write a flagged placeholder so the render survives.
    _write_bytes(output_path, _encode_solid_png(width, height, PLACEHOLDER_RGB))
    result = ImageResult(
        path=output_path,
        provider="placeholder",
        cost=0.0,
        is_placeholder=True,
        width=width,
        height=height,
    )
    if log:
        _log_image(
            result, prompt, aspect_ratio, quality, db_path, note="; ".join(errors)[:500]
        )
    return result
