"""
Text-to-speech for the production layer, behind a small provider interface.

Default provider is ``edge_tts`` (Microsoft Edge voices via the free, keyless
``edge-tts`` package). It yields both audio AND per-word timing boundaries
(WordBoundary events) so captions can be synced to the voiceover. ``elevenlabs``
is optional when a key is configured.

Everything is non-blocking: if synthesis fails (no network, missing package, no
key) the section falls back to a silent track with evenly-estimated word timings
so the render still proceeds.
"""

import os
import re
from typing import Callable, Optional

DEFAULT_VOICE = "en-US-GuyNeural"
_WPM_ESTIMATE = 155  # words/min for the silent fallback's timing spread


def _words(text: str) -> list:
    return re.findall(r"\S+", text or "")


def estimate_timings(text: str, start: float = 0.0, wpm: int = _WPM_ESTIMATE) -> list:
    """Evenly-spaced per-word timings (the fallback when no real boundaries)."""
    words = _words(text)
    if not words:
        return []
    per_word = 60.0 / max(1, wpm)
    timings = []
    cursor = start
    for word in words:
        timings.append({"word": word, "start": round(cursor, 3), "end": round(cursor + per_word, 3)})
        cursor += per_word
    return timings


def _edge_tts_synthesize(text: str, out_path: str, voice: str) -> dict:
    """Streams Edge TTS audio + WordBoundary timings to ``out_path`` (mp3)."""
    import asyncio

    import edge_tts

    async def _run() -> list:
        communicate = edge_tts.Communicate(text, voice or DEFAULT_VOICE)
        timings = []
        with open(out_path, "wb") as handle:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    handle.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7          # 100ns ticks -> seconds
                    duration = chunk["duration"] / 1e7
                    timings.append(
                        {"word": chunk["text"], "start": round(start, 3),
                         "end": round(start + duration, 3)}
                    )
        return timings

    timings = asyncio.run(_run())
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("edge_tts produced no audio")
    return {"audio_path": out_path, "word_timings": timings}


def _elevenlabs_synthesize(text: str, out_path: str, voice_id: str, api_key: str) -> dict:
    """ElevenLabs TTS (mp3). No word boundaries -> timings estimated by caller."""
    import requests

    if not api_key or not voice_id:
        raise RuntimeError("elevenlabs requires api_key + voice_id")
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=120,
    )
    response.raise_for_status()
    with open(out_path, "wb") as handle:
        handle.write(response.content)
    return {"audio_path": out_path, "word_timings": []}


def synthesize_section(
    text: str,
    out_path: str,
    cfg: Optional[dict] = None,
    voice: Optional[str] = None,
    synth: Optional[Callable] = None,
) -> dict:
    """
    Synthesizes one script section.

    Args:
        text (str): The narration text.
        out_path (str): Destination audio path (mp3).
        cfg (dict | None): TTS config (provider/voice/elevenlabs); loaded from
            config.py if omitted.
        voice (str | None): Voice override.
        synth: optional ``synth(text, out_path, voice) -> {audio_path,
            word_timings}`` used in tests so no real TTS call happens.

    Returns:
        result (dict): audio_path (None if silent fallback), word_timings (list),
        provider (str). Never raises.
    """
    if cfg is None:
        try:
            import sys

            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from config import get_tts_config

            cfg = get_tts_config()
        except Exception:
            cfg = {"provider": "edge_tts", "voice": DEFAULT_VOICE, "elevenlabs": {}}

    provider = (cfg.get("provider") or "edge_tts").strip()
    use_voice = voice or cfg.get("voice") or DEFAULT_VOICE
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        if synth is not None:
            result = synth(text, out_path, use_voice)
        elif provider == "elevenlabs":
            eleven = cfg.get("elevenlabs") or {}
            result = _elevenlabs_synthesize(
                text, out_path, eleven.get("voice_id", ""), eleven.get("api_key", "")
            )
        else:
            result = _edge_tts_synthesize(text, out_path, use_voice)

        timings = result.get("word_timings") or estimate_timings(text)
        return {
            "audio_path": result.get("audio_path"),
            "word_timings": timings,
            "provider": provider if synth is None else "injected",
        }
    except Exception:
        # Non-blocking fallback: silent track + estimated timings.
        return {
            "audio_path": None,
            "word_timings": estimate_timings(text),
            "provider": "estimated_silent",
        }
