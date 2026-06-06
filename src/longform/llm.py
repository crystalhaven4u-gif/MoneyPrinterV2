"""
Pluggable text-LLM client for the long-form creative layer.

Two providers behind one ``llm(prompt) -> str`` interface:

  * ``ollama``            -- local Ollama HTTP API (offline fallback, zero cost)
  * ``openai_compatible`` -- any OpenAI-style /chat/completions endpoint. Groq's
                             FREE tier is the recommended default for script
                             quality (base_url https://api.groq.com/openai/v1,
                             model llama-3.3-70b-versatile -- ~20x the local 3B).

The provider, model, and (for openai_compatible) base URL come from config; the
API key is read from an env var (never stored in config). ``make_llm`` keeps the
non-blocking fallback pattern: if the primary provider errors (e.g. no key, or a
network hiccup) it falls back to local Ollama rather than crashing a run.

This module also detects local Ollama models + system RAM and recommends the
largest viable local model -- but NEVER pulls weights on its own.
"""

import os
from typing import Callable, Optional

import requests

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def _load_config() -> dict:
    """Resolves the llm config from config.py, with safe offline defaults."""
    try:
        import sys

        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from config import get_llm_config, get_ollama_base_url, get_ollama_model

        cfg = get_llm_config()
        cfg["ollama_base_url"] = get_ollama_base_url() or DEFAULT_OLLAMA_BASE_URL
        cfg["ollama_model"] = get_ollama_model() or DEFAULT_OLLAMA_MODEL
        return cfg
    except Exception:
        return {
            "provider": "ollama",
            "model": "",
            "openai_compatible": {
                "base_url": DEFAULT_GROQ_BASE_URL,
                "api_key_env": "GROQ_API_KEY",
                "api_key": os.environ.get("GROQ_API_KEY", "").strip(),
            },
            "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
            "ollama_model": DEFAULT_OLLAMA_MODEL,
        }


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    session=None,
    temperature: float = 0.8,
    timeout: int = 300,
) -> str:
    """Generates text via the local Ollama ``/api/chat`` endpoint."""
    cfg = _load_config()
    base = (base_url or cfg.get("ollama_base_url") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    use_model = model or cfg.get("ollama_model") or DEFAULT_OLLAMA_MODEL
    http = session if session is not None else requests
    response = http.post(
        f"{base}/api/chat",
        json={
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return (body.get("message", {}).get("content", "") or "").strip()


def openai_compatible_generate(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
    session=None,
    temperature: float = 0.8,
    timeout: int = 120,
) -> str:
    """Generates text via an OpenAI-style ``/chat/completions`` endpoint."""
    if not api_key:
        raise RuntimeError("openai_compatible provider requires an API key")
    http = session if session is not None else requests
    response = http.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("openai_compatible response had no choices")
    return (choices[0].get("message", {}).get("content", "") or "").strip()


# --------------------------------------------------------------------------- #
# Dispatch + non-blocking fallback
# --------------------------------------------------------------------------- #
def make_llm(cfg: Optional[dict] = None, session=None, temperature: float = 0.8) -> Callable[[str], str]:
    """
    Returns an ``llm(prompt) -> str`` bound to the configured provider, with a
    non-blocking fallback to local Ollama if the primary provider errors.
    """
    cfg = cfg if cfg is not None else _load_config()
    provider = (cfg.get("provider") or "ollama").strip()
    oc = cfg.get("openai_compatible") or {}

    def _primary(prompt: str) -> str:
        if provider == "openai_compatible":
            return openai_compatible_generate(
                prompt,
                base_url=oc.get("base_url") or DEFAULT_GROQ_BASE_URL,
                model=cfg.get("model") or DEFAULT_GROQ_MODEL,
                api_key=oc.get("api_key") or os.environ.get(
                    oc.get("api_key_env", "GROQ_API_KEY"), ""
                ).strip(),
                session=session,
                temperature=temperature,
            )
        return ollama_generate(
            prompt, model=cfg.get("model") or None, session=session, temperature=temperature
        )

    def _llm(prompt: str) -> str:
        try:
            return _primary(prompt)
        except Exception:
            if provider != "ollama":
                # Non-blocking fallback: never crash a run on a remote failure.
                return ollama_generate(
                    prompt, model=cfg.get("model") if False else None,
                    session=session, temperature=temperature,
                )
            raise

    return _llm


def default_llm() -> Callable[[str], str]:
    """Returns an ``llm(prompt) -> str`` for the configured provider."""
    return make_llm()


def active_provider_label(cfg: Optional[dict] = None) -> str:
    """Human-readable description of which provider/model will actually run."""
    cfg = cfg if cfg is not None else _load_config()
    provider = (cfg.get("provider") or "ollama").strip()
    oc = cfg.get("openai_compatible") or {}
    if provider == "openai_compatible":
        has_key = bool(oc.get("api_key") or os.environ.get(oc.get("api_key_env", "GROQ_API_KEY"), ""))
        model = cfg.get("model") or DEFAULT_GROQ_MODEL
        if has_key:
            return f"openai_compatible ({model} @ {oc.get('base_url')})"
        return (
            f"openai_compatible ({model}) -- NO KEY in ${oc.get('api_key_env','GROQ_API_KEY')}; "
            f"will fall back to local Ollama"
        )
    return f"ollama ({cfg.get('model') or cfg.get('ollama_model') or DEFAULT_OLLAMA_MODEL})"


# --------------------------------------------------------------------------- #
# Local hardware detection + model recommendation (never auto-pulls)
# --------------------------------------------------------------------------- #
# Capable local models best-first, with an approximate RAM floor (GB) to run
# comfortably at q4. The floor per the engine spec is qwen2.5:7b-instruct /
# llama3.1:8b -- below that, script quality is not worth the trouble.
LOCAL_MODEL_LADDER = (
    ("llama3.1:70b", 40),
    ("qwen2.5:32b-instruct", 24),
    ("qwen2.5:14b-instruct", 12),
    ("llama3.1:8b", 8),
    ("qwen2.5:7b-instruct", 8),
)


def list_ollama_models(base_url: Optional[str] = None, session=None) -> list:
    """Lists locally installed Ollama models (``/api/tags``); [] on any error."""
    try:
        cfg = _load_config()
        base = (base_url or cfg.get("ollama_base_url") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        http = session if session is not None else requests
        response = http.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def system_ram_gb() -> Optional[float]:
    """Best-effort total system RAM in GB (psutil -> Windows API -> /proc)."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:  # Windows without psutil
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 1e9, 1)
    except Exception:
        pass
    try:  # Linux
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except Exception:
        pass
    return None


def recommend_local_model(ram_gb: Optional[float], installed: Optional[list] = None) -> dict:
    """
    Recommends the largest viable local model for the available RAM.

    Leaves ~30% headroom over a model's floor. Returns a dict with the
    recommendation, whether it is already installed, and whether a pull is
    needed (the caller must confirm before pulling -- this never pulls).
    """
    installed = installed or []
    installed_set = set(installed)
    floor_model, floor_ram = LOCAL_MODEL_LADDER[-1]

    if ram_gb is None:
        recommended, reason = floor_model, "RAM unknown; recommending the safe floor model"
    else:
        recommended, reason = None, ""
        for model, need in LOCAL_MODEL_LADDER:
            if ram_gb >= need / 0.7:  # 30% headroom
                recommended, reason = model, f"{ram_gb} GB RAM comfortably runs {model} (~{need} GB)"
                break
        if recommended is None:
            recommended = floor_model
            reason = f"{ram_gb} GB RAM is tight; floor model {floor_model} recommended (may be slow)"

    already = any(name == recommended or name.startswith(recommended + ":") for name in installed_set)
    return {
        "recommended": recommended,
        "reason": reason,
        "already_installed": already,
        "pull_needed": not already,
        "installed": installed,
        "ram_gb": ram_gb,
    }


def describe_local_setup(session=None) -> dict:
    """Bundles provider label, RAM, installed models, and the recommendation."""
    installed = list_ollama_models(session=session)
    ram = system_ram_gb()
    return {
        "active_provider": active_provider_label(),
        "ram_gb": ram,
        "installed_ollama_models": installed,
        "recommendation": recommend_local_model(ram, installed),
    }
