import os
import sys
import json
import srt_equalizer

from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])

def assert_folder_structure() -> None:
    """
    Make sure that the nessecary folder structure is present.

    Returns:
        None
    """
    # Create the .mp folder
    if not os.path.exists(os.path.join(ROOT_DIR, ".mp")):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {os.path.join(ROOT_DIR, '.mp')}", "green"))
        os.makedirs(os.path.join(ROOT_DIR, ".mp"))

def get_first_time_running() -> bool:
    """
    Checks if the program is running for the first time by checking if .mp folder exists.

    Returns:
        exists (bool): True if the program is running for the first time, False otherwise
    """
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))

def get_email_credentials() -> dict:
    """
    Gets the email credentials from the config file.

    Returns:
        credentials (dict): The email credentials
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["email"]

def get_verbose() -> bool:
    """
    Gets the verbose flag from the config file.

    Returns:
        verbose (bool): The verbose flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["verbose"]

def get_firefox_profile_path() -> str:
    """
    Gets the path to the Firefox profile.

    Returns:
        path (str): The path to the Firefox profile
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["firefox_profile"]

def get_headless() -> bool:
    """
    Gets the headless flag from the config file.

    Returns:
        headless (bool): The headless flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["headless"]

def get_ollama_base_url() -> str:
    """
    Gets the Ollama base URL.

    Returns:
        url (str): The Ollama base URL
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("ollama_base_url", "http://127.0.0.1:11434")

def get_ollama_model() -> str:
    """
    Gets the Ollama model name from the config file.

    Returns:
        model (str): The Ollama model name, or empty string if not set.
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("ollama_model", "")

def get_twitter_language() -> str:
    """
    Gets the Twitter language from the config file.

    Returns:
        language (str): The Twitter language
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["twitter_language"]

def get_nanobanana2_api_base_url() -> str:
    """
    Gets the Nano Banana 2 (Gemini image) API base URL.

    Returns:
        url (str): API base URL
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get(
            "nanobanana2_api_base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        )

def get_nanobanana2_api_key() -> str:
    """
    Gets the Nano Banana 2 API key.

    Returns:
        key (str): API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        configured = json.load(file).get("nanobanana2_api_key", "")
        return configured or os.environ.get("GEMINI_API_KEY", "")

def get_nanobanana2_model() -> str:
    """
    Gets the Nano Banana 2 model name.

    Returns:
        model (str): Model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("nanobanana2_model", "gemini-3.1-flash-image-preview")

def get_nanobanana2_aspect_ratio() -> str:
    """
    Gets the aspect ratio for Nano Banana 2 image generation.

    Returns:
        ratio (str): Aspect ratio
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("nanobanana2_aspect_ratio", "9:16")

def get_threads() -> int:
    """
    Gets the amount of threads to use for example when writing to a file with MoviePy.

    Returns:
        threads (int): Amount of threads
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["threads"]
    
def get_zip_url() -> str:
    """
    Gets the URL to the zip file containing the songs.

    Returns:
        url (str): The URL to the zip file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["zip_url"]

def get_is_for_kids() -> bool:
    """
    Gets the is for kids flag from the config file.

    Returns:
        is_for_kids (bool): The is for kids flag
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["is_for_kids"]

def get_google_maps_scraper_zip_url() -> str:
    """
    Gets the URL to the zip file containing the Google Maps scraper.

    Returns:
        url (str): The URL to the zip file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["google_maps_scraper"]

def get_google_maps_scraper_niche() -> str:
    """
    Gets the niche for the Google Maps scraper.

    Returns:
        niche (str): The niche
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["google_maps_scraper_niche"]

def get_scraper_timeout() -> int:
    """
    Gets the timeout for the scraper.

    Returns:
        timeout (int): The timeout
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["scraper_timeout"] or 300

def get_outreach_message_subject() -> str:
    """
    Gets the outreach message subject.

    Returns:
        subject (str): The outreach message subject
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["outreach_message_subject"]
    
def get_outreach_message_body_file() -> str:
    """
    Gets the outreach message body file.

    Returns:
        file (str): The outreach message body file
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["outreach_message_body_file"]

def get_tts_voice() -> str:
    """
    Gets the TTS voice from the config file.

    Returns:
        voice (str): The TTS voice
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("tts_voice", "Jasper")

def get_assemblyai_api_key() -> str:
    """
    Gets the AssemblyAI API key.

    Returns:
        key (str): The AssemblyAI API key
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["assembly_ai_api_key"]

def get_stt_provider() -> str:
    """
    Gets the configured STT provider.

    Returns:
        provider (str): The STT provider
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("stt_provider", "local_whisper")

def get_whisper_model() -> str:
    """
    Gets the local Whisper model name.

    Returns:
        model (str): Whisper model name
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("whisper_model", "base")

def get_whisper_device() -> str:
    """
    Gets the target device for Whisper inference.

    Returns:
        device (str): Whisper device
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("whisper_device", "auto")

def get_whisper_compute_type() -> str:
    """
    Gets the compute type for Whisper inference.

    Returns:
        compute_type (str): Whisper compute type
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("whisper_compute_type", "int8")
    
def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    """
    Equalizes the subtitles in a SRT file.

    Args:
        srt_path (str): The path to the SRT file
        max_chars (int): The maximum amount of characters in a subtitle

    Returns:
        None
    """
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
    
def get_font() -> str:
    """
    Gets the font from the config file.

    Returns:
        font (str): The font
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["font"]

def get_fonts_dir() -> str:
    """
    Gets the fonts directory.

    Returns:
        dir (str): The fonts directory
    """
    return os.path.join(ROOT_DIR, "fonts")

def get_imagemagick_path() -> str:
    """
    Gets the path to ImageMagick.

    Returns:
        path (str): The path to ImageMagick
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["imagemagick_path"]

def get_script_sentence_length() -> int:
    """
    Gets the forced script's sentence length.
    In case there is no sentence length in config, returns 4 when none

    Returns:
        length (int): Length of script's sentence
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        config_json = json.load(file)
        if (config_json.get("script_sentence_length") is not None):
            return config_json["script_sentence_length"]
        else:
            return 4

def get_post_bridge_config() -> dict:
    """
    Gets the Post Bridge configuration with safe defaults.

    Returns:
        config (dict): Sanitized Post Bridge configuration
    """
    defaults = {
        "enabled": False,
        "api_key": "",
        "platforms": ["tiktok", "instagram"],
        "account_ids": [],
        "auto_crosspost": False,
    }
    supported_platforms = {"tiktok", "instagram"}

    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        config_json = json.load(file)

    raw_config = config_json.get("post_bridge", {})
    if not isinstance(raw_config, dict):
        raw_config = {}

    raw_platforms = raw_config.get("platforms")
    normalized_platforms = []
    seen_platforms = set()

    if raw_platforms is None:
        normalized_platforms = defaults["platforms"].copy()
    elif isinstance(raw_platforms, list):
        for platform in raw_platforms:
            normalized_platform = str(platform).strip().lower()
            if (
                normalized_platform in supported_platforms
                and normalized_platform not in seen_platforms
            ):
                normalized_platforms.append(normalized_platform)
                seen_platforms.add(normalized_platform)
    else:
        normalized_platforms = []

    raw_account_ids = raw_config.get("account_ids", defaults["account_ids"])
    normalized_account_ids = []
    if isinstance(raw_account_ids, list):
        for account_id in raw_account_ids:
            try:
                normalized_account_ids.append(int(account_id))
            except (TypeError, ValueError):
                continue

    api_key = str(raw_config.get("api_key", "")).strip()
    if not api_key:
        api_key = os.environ.get("POST_BRIDGE_API_KEY", "").strip()

    return {
        "enabled": bool(raw_config.get("enabled", defaults["enabled"])),
        "api_key": api_key,
        "platforms": normalized_platforms,
        "account_ids": normalized_account_ids,
        "auto_crosspost": bool(
            raw_config.get("auto_crosspost", defaults["auto_crosspost"])
        ),
    }

def _get_longform_config() -> dict:
    """
    Reads the "longform" config block (safe default empty dict).

    Returns:
        config (dict): The longform configuration block.
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("longform", {})
    return raw if isinstance(raw, dict) else {}

def get_longform_youtube_api_key() -> str:
    """
    Gets the YouTube Data API key for the long-form data farmer.

    The YOUTUBE_API_KEY environment variable takes precedence; otherwise the
    value of longform.youtube_api_key in config.json is used. This is the READ
    API key and is intentionally separate from the OAuth upload client.

    Returns:
        key (str): The API key, or an empty string if unset.
    """
    env_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if env_key:
        return env_key
    return str(_get_longform_config().get("youtube_api_key", "")).strip()

def get_longform_daily_quota_budget() -> int:
    """
    Gets the daily YouTube Data API quota budget for the farmer.

    Returns:
        budget (int): Quota units to spend per day (default 9000 of the
            10000/day free tier).
    """
    try:
        return int(_get_longform_config().get("daily_quota_budget", 9000))
    except (TypeError, ValueError):
        return 9000

def _get_image_config() -> dict:
    """
    Reads the "image" config block (safe default empty dict).

    Returns:
        config (dict): The image-provider configuration block.
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("image", {})
    return raw if isinstance(raw, dict) else {}

def get_image_config() -> dict:
    """
    Gets the fully-resolved image-provider configuration for the long-form
    engine, with safe defaults so the free path works with zero setup.

    The Gemini sub-block reuses the existing Nano Banana 2 (Gemini) credentials
    rather than duplicating a key -- there is only ever one Gemini key.

    Returns:
        config (dict): default_provider, thumbnail_provider, fallback_order,
            cloudflare {account_id, api_token}, local_sd_url, gemini_quality,
            and a resolved gemini {api_key, base_url, model} sub-block.
    """
    raw = _get_image_config()
    cloudflare = raw.get("cloudflare") or {}
    pollinations = raw.get("pollinations") or {}
    default_provider = str(raw.get("default_provider", "pollinations") or "pollinations")
    return {
        "default_provider": default_provider,
        "thumbnail_provider": str(
            raw.get("thumbnail_provider", default_provider) or default_provider
        ),
        "fallback_order": list(
            raw.get("fallback_order") or ["pollinations", "cloudflare", "gemini"]
        ),
        "pollinations": {
            "referrer": str(pollinations.get("referrer", "") or "").strip(),
            "token": str(pollinations.get("token", "") or "").strip(),
        },
        "cloudflare": {
            "account_id": str(cloudflare.get("account_id", "") or "").strip(),
            "api_token": str(cloudflare.get("api_token", "") or "").strip(),
        },
        "local_sd_url": str(raw.get("local_sd_url", "") or "").strip(),
        "gemini_quality": str(raw.get("gemini_quality", "standard") or "standard"),
        "gemini": {
            "api_key": get_nanobanana2_api_key(),
            "base_url": get_nanobanana2_api_base_url(),
            "model": get_nanobanana2_model(),
        },
    }

def _get_llm_config() -> dict:
    """Reads the "llm" config block (safe default empty dict)."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("llm", {})
    return raw if isinstance(raw, dict) else {}

def get_llm_config() -> dict:
    """
    Gets the resolved LLM configuration for the long-form creative layer.

    provider is 'ollama' (local, offline fallback) or 'openai_compatible' (any
    OpenAI-style /chat/completions endpoint -- Groq's FREE tier is the
    recommended default for script quality). The API key is read from the env
    var named by openai_compatible.api_key_env (never stored in config). If
    llm.model is empty, the provider-specific default is used.

    Returns:
        config (dict): provider, model, and an openai_compatible
            {base_url, api_key_env, api_key} sub-block.
    """
    raw = _get_llm_config()
    oc = raw.get("openai_compatible") or {}
    api_key_env = str(oc.get("api_key_env", "GROQ_API_KEY") or "GROQ_API_KEY").strip()
    return {
        "provider": str(raw.get("provider", "ollama") or "ollama").strip(),
        "model": str(raw.get("model", "") or "").strip(),
        "openai_compatible": {
            "base_url": str(
                oc.get("base_url", "https://api.groq.com/openai/v1")
                or "https://api.groq.com/openai/v1"
            ).strip(),
            "api_key_env": api_key_env,
            # Env var takes precedence; otherwise an inline api_key in the
            # (gitignored) config is used so the key can persist locally.
            "api_key": os.environ.get(api_key_env, "").strip()
            or str(oc.get("api_key", "") or "").strip(),
        },
    }

def _get_script_config() -> dict:
    """Reads the "script" config block (safe default empty dict)."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("script", {})
    return raw if isinstance(raw, dict) else {}

def get_script_grounding() -> bool:
    """Whether per-topic factual grounding (web research) is enabled."""
    value = _get_script_config().get("grounding", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)

def get_script_narrative() -> bool:
    """Whether Pass 2 (cinematic narrative rewrite) is enabled (default on)."""
    value = _get_script_config().get("narrative", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)

def get_script_target_minutes() -> int:
    """Target runtime in minutes for generated long-form scripts."""
    try:
        return int(_get_script_config().get("target_minutes", 10))
    except (TypeError, ValueError):
        return 10

def get_ffmpeg_path() -> str:
    """
    Path to the ffmpeg binary for the production layer.

    Prefers config.json ffmpeg_path, then the FFMPEG_PATH env var, then bare
    "ffmpeg" (resolved on PATH by the caller).
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        value = json.load(file).get("ffmpeg_path", "")
    return str(value or os.environ.get("FFMPEG_PATH", "") or "ffmpeg").strip()

def _get_tts_config() -> dict:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("tts", {})
    return raw if isinstance(raw, dict) else {}

def get_tts_config() -> dict:
    """
    Resolved TTS config for the production layer.

    provider is 'edge_tts' (free, no key — default) or 'elevenlabs' (needs a key
    from the env var named by elevenlabs.api_key_env). voice is provider-specific.
    """
    raw = _get_tts_config()
    eleven = raw.get("elevenlabs") or {}
    api_key_env = str(eleven.get("api_key_env", "ELEVENLABS_API_KEY") or "ELEVENLABS_API_KEY").strip()
    return {
        "provider": str(raw.get("provider", "edge_tts") or "edge_tts").strip(),
        "voice": str(raw.get("voice", "en-US-GuyNeural") or "en-US-GuyNeural").strip(),
        "elevenlabs": {
            "api_key_env": api_key_env,
            "api_key": os.environ.get(api_key_env, "").strip()
            or str(eleven.get("api_key", "") or "").strip(),
            "voice_id": str(eleven.get("voice_id", "") or "").strip(),
        },
    }

def _get_footage_config() -> dict:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        raw = json.load(file).get("footage", {})
    return raw if isinstance(raw, dict) else {}

def get_footage_config() -> dict:
    """
    Resolved footage-sourcer config. Stock-API keys are read from env vars named
    by *_api_key_env (keyless sources need nothing). music_dir holds local
    royalty-free beds.
    """
    raw = _get_footage_config()
    pexels_env = str(raw.get("pexels_api_key_env", "PEXELS_API_KEY") or "PEXELS_API_KEY").strip()
    pixabay_env = str(raw.get("pixabay_api_key_env", "PIXABAY_API_KEY") or "PIXABAY_API_KEY").strip()
    return {
        "pexels_api_key_env": pexels_env,
        "pexels_api_key": os.environ.get(pexels_env, "").strip()
        or str(raw.get("pexels_api_key", "") or "").strip(),
        "pixabay_api_key_env": pixabay_env,
        "pixabay_api_key": os.environ.get(pixabay_env, "").strip()
        or str(raw.get("pixabay_api_key", "") or "").strip(),
        "music_dir": str(raw.get("music_dir", "assets/music") or "assets/music").strip(),
        "music_mood": str(raw.get("music_mood", "dark ambient") or "dark ambient").strip(),
    }
