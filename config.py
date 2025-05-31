import os
from pathlib import Path # Added for Path object
from dotenv import load_dotenv

load_dotenv()

# API Keys
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# API URLs and Model IDs
REPLICATE_API_URL = os.getenv("REPLICATE_API_URL", "https://api.replicate.com/v1/predictions")
REPLICATE_ZEROSCOPE_MODEL_XL = os.getenv("REPLICATE_ZEROSCOPE_MODEL_XL", "anotherjesse/zeroscope-v2-xl:9f7476737190e1a712580adfd5446408f14b2de0e6e8e168d68f2029fc221216")
REPLICATE_ZEROSCOPE_MODEL_576W = os.getenv("REPLICATE_ZEROSCOPE_MODEL_576W", "anotherjesse/zeroscope-v2-576w:1c8f6c34d800a8054187871f754559085323598320e960e699500244a8386153")
STABILITY_TEXT_TO_IMAGE_API_URL_BASE = os.getenv("STABILITY_TEXT_TO_IMAGE_API_URL_BASE", "https://api.stability.ai/v1/generation/{engine_id}/text-to-image")
STABILITY_DEFAULT_ENGINE = os.getenv("STABILITY_DEFAULT_ENGINE", "stable-diffusion-xl-1024-v1-0")
STABILITY_AUDIO_API_URL = os.getenv("STABILITY_AUDIO_API_URL", "https://api.stability.ai/v1/generation/stable-audio-generate-v1")

# Default video/audio settings
DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "24")) # Ensure integer
REPLICATE_POLL_INTERVAL = int(os.getenv("REPLICATE_POLL_INTERVAL", "10")) # seconds
REPLICATE_API_TIMEOUT = float(os.getenv("REPLICATE_API_TIMEOUT", "300.0")) # seconds
STABILITY_API_TIMEOUT = float(os.getenv("STABILITY_API_TIMEOUT", "180.0")) # seconds
ELEVENLABS_API_TIMEOUT = float(os.getenv("ELEVENLABS_API_TIMEOUT", "60.0")) # seconds
ASSEMBLYAI_POLL_INTERVAL = int(os.getenv("ASSEMBLYAI_POLL_INTERVAL", "5")) # seconds
ASSEMBLYAI_API_TIMEOUT = float(os.getenv("ASSEMBLYAI_API_TIMEOUT", "300.0")) # seconds
REPLICATE_MAX_POLL_ATTEMPTS = int(os.getenv("REPLICATE_MAX_POLL_ATTEMPTS", "30"))

# File paths and commands
TEMP_DIR_BASE = os.getenv("TEMP_DIR_BASE", "temp_files")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
STATIC_VIDEO_DIR_BASE = Path(STATIC_DIR) / "generated_videos"
FFMPEG_COMMAND = os.getenv("FFMPEG_COMMAND", "ffmpeg")

# Feature flags
CLEANUP_TEMP_FILES = os.getenv("CLEANUP_TEMP_FILES", "True").lower() == "true"


# Voice mapping for ElevenLabs
ELEVENLABS_LANG_VOICE_MAP = {
    "ja": "hBWDuZMNs32sP5dKzMuc",      # Japanese
    "en": "21m00Tcm4TlvDq8ikA2E",      # English (Example, replace with a preferred English voice ID)
    "it": "fzDFBB4mgvMlL36gPXcz",      # Italian
    "es": "0vrPGvXHhDD3rbGURCk8",      # Spanish
    "fr": "iRYhWuT8tKZ81GesmMsh",      # French
    "de": "sx7WD8TJIOrk5RQOptDH",      # German
    "zh": "4VZIsMPtgggwNg7OXbPY",      # Chinese
    "ko": "WqVy7827vjE2r3jWvbnP",      # Korean
    # Add other languages and corresponding voice IDs as needed
}
# Attempt to load from JSON environment variable if present
elevenlabs_map_json = os.getenv("ELEVENLABS_LANG_VOICE_MAP_JSON")
if elevenlabs_map_json:
    try:
        ELEVENLABS_LANG_VOICE_MAP.update(json.loads(elevenlabs_map_json))
    except json.JSONDecodeError:
        print("Warning: Could not parse ELEVENLABS_LANG_VOICE_MAP_JSON from environment.")


# DeepL Language Codes (ensure these match DeepL's expected format)
DEEPL_LANG_MAP = {
    "en": "EN-US",
    "ja": "JA",
    "es": "ES",
    "fr": "FR",
    "de": "DE",
    "it": "IT",
    "zh": "ZH",
}
# Attempt to load from JSON environment variable if present
deepl_map_json = os.getenv("DEEPL_LANG_MAP_JSON")
if deepl_map_json:
    try:
        DEEPL_LANG_MAP.update(json.loads(deepl_map_json))
    except json.JSONDecodeError:
        print("Warning: Could not parse DEEPL_LANG_MAP_JSON from environment.")
