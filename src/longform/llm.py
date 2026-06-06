"""
Minimal Ollama text client for the long-form creative layer.

Talks to the local Ollama HTTP API with ``requests`` instead of the ``ollama``
SDK, so the long-form package keeps its light dependency footprint and does not
import the Shorts-side ``llm_provider``. Model and base URL are resolved lazily
from config (with env fallbacks), and every generator accepts an injectable
``llm`` callable so tests never touch a real server.
"""

import os
from typing import Callable, Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"


def _resolve_config() -> tuple:
    """Returns (base_url, model) from config.py, falling back to env/defaults."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    if base_url and model:
        return base_url, model
    try:
        import sys

        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from config import get_ollama_base_url, get_ollama_model

        return (base_url or get_ollama_base_url() or DEFAULT_BASE_URL,
                model or get_ollama_model() or DEFAULT_MODEL)
    except Exception:
        return base_url or DEFAULT_BASE_URL, model or DEFAULT_MODEL


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    session=None,
    temperature: float = 0.8,
    timeout: int = 300,
) -> str:
    """
    Generates text via the local Ollama ``/api/chat`` endpoint.

    Args:
        prompt (str): The user prompt.
        model / base_url (str | None): Overrides; resolved from config if unset.
        session: A ``requests``-like object exposing ``.post``; defaults to
            ``requests`` (injected in tests).
        temperature (float): Sampling temperature.
        timeout (int): Request timeout in seconds.

    Returns:
        text (str): The model's reply, stripped.
    """
    resolved_base, resolved_model = _resolve_config()
    base = (base_url or resolved_base).rstrip("/")
    http = session if session is not None else requests
    response = http.post(
        f"{base}/api/chat",
        json={
            "model": model or resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return (body.get("message", {}).get("content", "") or "").strip()


def default_llm() -> Callable[[str], str]:
    """Returns a one-arg ``llm(prompt) -> str`` bound to the configured Ollama."""
    return lambda prompt: ollama_generate(prompt)
