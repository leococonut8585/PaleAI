
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") # Added AssemblyAI API Key
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") # Added DeepL API Key

# Default video settings
DEFAULT_FPS = 25
REPLICATE_POLL_INTERVAL = 10 # seconds
REPLICATE_API_TIMEOUT = 300.0 # seconds (increased for potentially longer video processing)
STABILITY_API_TIMEOUT = 120.0 # seconds
ELEVENLABS_API_TIMEOUT = 60.0 # seconds
ASSEMBLYAI_POLL_INTERVAL = 5 # seconds
ASSEMBLYAI_API_TIMEOUT = 300.0 # seconds for transcription

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

# DeepL Language Codes (ensure these match DeepL's expected format)
DEEPL_LANG_MAP = {
    "en": "EN-US", # Corrected to EN-US for compatibility with DeepL
    "ja": "JA",
    "es": "ES",
    "fr": "FR",
    "de": "DE",
    "it": "IT",
    "zh": "ZH",
    # Add more as needed
}
```
