```python
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import logging
import os
import re
import uuid
import asyncio
import httpx
from pathlib import Path
import shutil
import base64
import random # For BGM seed
from enum import Enum 

from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip, ImageClip, 
    concatenate_videoclips, TextClip, CompositeVideoClip
)
from moviepy.video.fx.all import resize as moviepy_resize 
from moviepy.config import change_settings
from PIL import Image 

# Import config variables from config.py
try:
    from config import (
        REPLICATE_API_TOKEN, STABILITY_API_KEY, ELEVENLABS_API_KEY, ASSEMBLYAI_API_KEY, DEEPL_API_KEY,
        REPLICATE_API_URL, REPLICATE_ZEROSCOPE_MODEL_XL, REPLICATE_ZEROSCOPE_MODEL_576W,
        STABILITY_TEXT_TO_IMAGE_API_URL_BASE,
        STABILITY_DEFAULT_ENGINE, STABILITY_AUDIO_API_URL, REPLICATE_POLL_INTERVAL,
        REPLICATE_API_TIMEOUT, STABILITY_API_TIMEOUT, ELEVENLABS_API_TIMEOUT,
        ASSEMBLYAI_POLL_INTERVAL, ASSEMBLYAI_API_TIMEOUT, DEFAULT_FPS,
        TEMP_DIR_BASE, STATIC_VIDEO_DIR_BASE, CLEANUP_TEMP_FILES, ELEVENLABS_LANG_VOICE_MAP, DEEPL_LANG_MAP,
        REPLICATE_MAX_POLL_ATTEMPTS, FFMPEG_COMMAND
    )
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import from config.py. Ensure the file exists and variables are defined. Using fallback defaults.")
    # Provide fallback defaults if config is missing, useful for basic testing without .env
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

    REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
    REPLICATE_ZEROSCOPE_MODEL_XL = "anotherjesse/zeroscope-v2-xl:9f7476737190e1a712580adfd5446408f14b2de0e6e8e168d68f2029fc221216"
    REPLICATE_ZEROSCOPE_MODEL_576W = "anotherjesse/zeroscope-v2-576w:1c8f6c34d800a8054187871f754559085323598320e960e699500244a8386153" 
    STABILITY_TEXT_TO_IMAGE_API_URL_BASE = "https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
    STABILITY_DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0"
    STABILITY_AUDIO_API_URL = "https://api.stability.ai/v1/generation/stable-audio-generate-v1" 
    REPLICATE_POLL_INTERVAL = 10
    REPLICATE_API_TIMEOUT = 300.0
    STABILITY_API_TIMEOUT = 180.0 
    ELEVENLABS_API_TIMEOUT = 60.0
    ASSEMBLYAI_POLL_INTERVAL = 5
    ASSEMBLYAI_API_TIMEOUT = 300.0
    DEFAULT_FPS = 24 # Changed
    TEMP_DIR_BASE = "temp_files"
    STATIC_VIDEO_DIR_BASE = Path("./static/generated_videos")
    CLEANUP_TEMP_FILES = True
    ELEVENLABS_LANG_VOICE_MAP = {"en": "21m00Tcm4TlvDq8ikA2E", "ja": "SOYHLrjzK2X1ezoPC6cr"}
    DEEPL_LANG_MAP = {"en": "EN-US", "ja": "JA"}
    REPLICATE_MAX_POLL_ATTEMPTS = 30
    FFMPEG_COMMAND = "ffmpeg"


# ImageMagickのパスを設定（環境変数または直接パスで指定）
imagemagick_path = os.getenv("IMAGEMAGICK_BINARY")
if imagemagick_path:
    change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
else:
    logging.warning("IMAGEMAGICK_BINARY environment variable not set. TextClip might not work correctly.")


app = FastAPI() # This will be replaced by the app instance from main.py if imported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(TEMP_DIR_BASE)
STATIC_VIDEO_DIR = Path(STATIC_VIDEO_DIR_BASE)
STATIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# --- Pydantic Models ---
class EditTargetEnum(str, Enum):
    SCENE_VIDEO = "scene_video"
    SCENE_DURATION = "scene_duration"
    NARRATION_SEGMENT_TEXT = "narration_segment_text"
    NARRATION_SEGMENT_VOICE = "narration_segment_voice"
    NARRATION_FULL_SCRIPT = "narration_full_script"
    SUBTITLE_ENTRY_TEXT = "subtitle_entry_text" 
    SUBTITLE_FULL_SCRIPT = "subtitle_full_script"
    SUBTITLE_STYLE = "subtitle_style" 
    BGM = "bgm"

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Overall theme or story for the video.")
    duration: Optional[int] = Field(10, ge=1, le=600, description="Target total duration of the video in seconds. Claude will try to match this.")
    resolution: str = Field("1024x576", description="Resolution, e.g. '512x512', '1920x1080'. Will be adapted for specific APIs.")
    scene_prompts: Optional[List[str]] = Field(None, description="Optional list of detailed prompts for each scene.")
    
    narration_enabled: bool = Field(True, description="Enable or disable narration generation.")
    narration_script_prompt: Optional[str] = Field(None, description="Prompt for Claude to generate narration script.")
    narration_lang: str = Field("en", description="Language for narration (e.g., JA, EN).")
    narration_voice_id: Optional[str] = Field(None, description="Specific ElevenLabs voice ID. If None, a default for the language is used.")
    
    subtitles_enabled: bool = Field(True, description="Enable or disable subtitle generation.")
    subtitle_script_prompt: Optional[str] = Field(None, description="Prompt for Claude to generate subtitle texts.")
    subtitle_source_lang: Optional[str] = Field(None, description="Source language of subtitles if translation is needed, defaults to narration_lang.")
    subtitle_target_lang: Optional[str] = Field(None, description="Target language for subtitle translation (e.g., EN, ES).")
    subtitle_font_name: Optional[str] = Field(None, description="Subtitle font name (e.g., 'Arial', 'Meiryo')")
    subtitle_font_size: Optional[str] = Field(None, description="Subtitle font size (e.g., '24')")
    subtitle_primary_color: Optional[str] = Field(None, description="Subtitle primary color (e.g., '&H00FFFFFF' for white, '&H000000FF' for red in ASS/SSA hex format)")
    subtitle_outline_color: Optional[str] = Field(None, description="Subtitle outline color (e.g., '&H00000000' for black)")
    subtitle_background_color: Optional[str] = Field(None, description="Subtitle background/box color (e.g., '&H80000000' for semi-transparent black)")
    subtitle_alignment: Optional[int] = Field(None, description="Subtitle alignment (Numpad notation: 1-bottom-left, 2-bottom-center, ..., 9-top-right)")
    subtitle_margin_v: Optional[int] = Field(None, description="Subtitle vertical margin from edge of video (pixels)")
    
    bgm_enabled: bool = Field(True, description="Enable or disable background music.")
    bgm_prompt: Optional[str] = Field(None, description="Prompt describing the desired BGM.")
    
    output_format: str = Field("mp4", description="Output video format (e.g., mp4, mov, webm).")
    video_quality: str = Field("medium", description="Desired video quality (e.g., low, medium, high). This can influence parameters like num_inference_steps.")

    replicate_prompt_prefix: Optional[str] = Field(None, description="Prefix added to the main scene prompt for Replicate generation. e.g., 'cinematic, detailed, ...'")
    replicate_negative_prompt: Optional[str] = Field("blurry, low quality, worst quality, low resolution, text, watermark, bad anatomy, artifacts, deformed, ugly, noise, grain", description="Negative prompt for Replicate generation.")
    replicate_guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for Replicate (e.g., 7.5). Higher values mean stricter prompt adherence.")
    replicate_num_inference_steps: Optional[int] = Field(25, ge=10, le=100, description="Number of inference steps for Replicate (e.g., 25). Higher values can improve quality but increase time.")
    replicate_seed: Optional[int] = Field(None, description="Seed for Replicate generation for reproducibility. If None or -1, random.")
    replicate_model_version: Optional[str] = Field(None, description=f"Specify Replicate model version. Default: Zeroscope v2 XL ('{REPLICATE_ZEROSCOPE_MODEL_XL}'). Alternative: Zeroscope v2 576w ('{REPLICATE_ZEROSCOPE_MODEL_576W}')")


class VideoGenerationResponse(BaseModel):
    message: str
    task_id: str
    status_url: str
    video_url: Optional[str] = None
    debug_data_url: Optional[str] = None
    error: Optional[str] = None

class VideoEditRequest(BaseModel):
    edit_target: EditTargetEnum = Field(..., description="The type of element to be edited.")
    target_indices: Optional[List[int]] = Field(None, 
        description="List of 0-based indices for the target element (e.g., scene number(s), narration segment number(s)). Interpretation depends on edit_target.")
    new_prompt: Optional[str] = Field(None, 
        description="New textual prompt (e.g., for scene description, narration segment, BGM description).")
    new_value_numeric: Optional[float] = Field(None, 
        description="New numerical value (e.g., scene duration in seconds).")
    new_value_string: Optional[str] = Field(None, 
        description="New string value (e.g., voice ID for narration, specific model ID).")
    new_script_srt: Optional[str] = Field(None, 
        description="New full subtitle script in SRT format. Used when edit_target is SUBTITLE_FULL_SCRIPT.")
    
    # Fields from VideoGenerationRequest that might be updated during an edit
    prompt: Optional[str] = None
    duration: Optional[int] = Field(None, ge=1, le=600)
    resolution: Optional[str] = None
    scene_prompts: Optional[List[str]] = None
    narration_enabled: Optional[bool] = None
    narration_script_prompt: Optional[str] = None
    narration_lang: Optional[str] = None
    narration_voice_id: Optional[str] = None
    subtitles_enabled: Optional[bool] = None
    subtitle_source_lang: Optional[str] = None
    subtitle_target_lang: Optional[str] = None
    subtitle_font_name: Optional[str] = None
    subtitle_font_size: Optional[str] = None
    subtitle_primary_color: Optional[str] = None
    subtitle_outline_color: Optional[str] = None
    subtitle_background_color: Optional[str] = None
    subtitle_alignment: Optional[int] = None
    subtitle_margin_v: Optional[int] = None
    bgm_enabled: Optional[bool] = None
    bgm_prompt: Optional[str] = None
    output_format: Optional[str] = None
    video_quality: Optional[str] = None
    replicate_prompt_prefix: Optional[str] = None
    replicate_negative_prompt: Optional[str] = None
    replicate_guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0)
    replicate_num_inference_steps: Optional[int] = Field(None, ge=10, le=100)
    replicate_seed: Optional[int] = None
    replicate_model_version: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[int] = None
    result_url: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


def save_status(task_id: str, data: dict):
    status_file = TEMP_DIR / task_id / f"{task_id}_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True) 
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving status for task {task_id} to {status_file}: {e}")

def get_status_from_file(task_id: str) -> Optional[Dict[str, Any]]: 
    status_file = TEMP_DIR / task_id / f"{task_id}_status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding status file for task {task_id}")
            return None
    return None

def get_default_voice_id(lang: str) -> str:
    try:
        from config import ELEVENLABS_LANG_VOICE_MAP
        return ELEVENLABS_LANG_VOICE_MAP.get(lang.lower(), ELEVENLABS_LANG_VOICE_MAP.get("en", "21m00Tcm4TlvDq8ikA2E"))
    except ImportError:
        logger.warning("config.py not found or ELEVENLABS_LANG_VOICE_MAP not defined. Using a hardcoded default.")
        fallback_map = {"en": "21m00Tcm4TlvDq8ikA2E", "ja": "SOYHLrjzK2X1ezoPC6cr"}
        return fallback_map.get(lang.lower(), fallback_map["en"])

def parse_resolution(resolution_str: str) -> (int, int):
    try:
        width, height = map(int, resolution_str.split('x'))
        return width, height
    except ValueError:
        logger.warning(f"Invalid resolution format: '{resolution_str}'. Falling back to 1024x576.")
        return 1024, 576

# Note for developers: FFmpeg needs access to fonts for subtitle rendering.
# Ensure that necessary fonts (e.g., Arial, Noto Sans JP for Japanese) are installed on the system
# where FFmpeg is executed. Alternatively, you can specify a font directory using the
# `fontsdir` option in the subtitles filter (e.g., `subtitles=filename=/path/to/subs.srt:fontsdir=/path/to/fonts`)
# or set the `FC_CONFIG_DIR` or `FONTCONFIG_PATH` environment variables if Fontconfig is used by your FFmpeg build.
async def run_ffmpeg_command_async(command: List[str], request_id: str) -> tuple[bool, str, str]:
    cmd_str = " ".join(command)
    logger.info(f"[{request_id}] FFmpeg: Running command: {cmd_str}")
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        stdout_str = stdout.decode(errors='ignore') if stdout else ""
        stderr_str = stderr.decode(errors='ignore') if stderr else ""

        if process.returncode == 0:
            logger.info(f"[{request_id}] FFmpeg command executed successfully.")
            if stderr_str: # FFmpeg often outputs info to stderr
                logger.info(f"[{request_id}] FFmpeg STDERR (Info):\n{stderr_str}")
            return True, stdout_str, stderr_str
        else:
            logger.error(f"[{request_id}] FFmpeg command failed with return code {process.returncode}.")
            logger.error(f"[{request_id}] FFmpeg STDOUT:\n{stdout_str}")
            logger.error(f"[{request_id}] FFmpeg STDERR:\n{stderr_str}")
            return False, stdout_str, stderr_str
    except FileNotFoundError:
        logger.error(f"[{request_id}] FFmpeg command 'ffmpeg' not found. Please ensure FFmpeg is installed and in PATH.")
        return False, "", "FFmpeg command not found."
    except Exception as e:
        logger.error(f"[{request_id}] An exception occurred while running FFmpeg command: {e}", exc_info=True)
        return False, "", str(e)

def format_time_srt(seconds: float) -> str:
    """Helper function to format time in SRT format (HH:MM:SS,mmm)"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def _format_assemblyai_transcript_to_srt(transcript_data: Dict, max_line_len: int = 40, max_line_duration_s: float = 7.0, min_line_duration_s: float = 1.0) -> str:
    words = transcript_data.get("words")
    if not words:
        return ""

    srt_entries = []
    srt_idx = 1
    current_line_text = ""
    current_line_start_time_ms = -1
    
    for i, word_info in enumerate(words):
        word_text = word_info["text"]
        word_start_ms = word_info["start"]
        word_end_ms = word_info["end"]

        if current_line_start_time_ms == -1: 
            current_line_text = word_text
            current_line_start_time_ms = word_start_ms
        else:
            potential_line = current_line_text + " " + word_text
            potential_duration_s = (word_end_ms - current_line_start_time_ms) / 1000.0
            
            break_line = False
            if len(potential_line) > max_line_len:
                break_line = True
            elif potential_duration_s > max_line_duration_s:
                break_line = True
            elif word_text.endswith(('.', '?', '!')) : 
                break_line = True
            elif i + 1 < len(words) and (words[i+1]["start"] - word_end_ms) > 500: 
                break_line = True
            
            if break_line:
                line_end_time_ms = words[i-1]["end"] 
                line_duration_s = (line_end_time_ms - current_line_start_time_ms) / 1000.0
                if line_duration_s < min_line_duration_s:
                    line_end_time_ms = current_line_start_time_ms + int(min_line_duration_s * 1000)
                    if i < len(words) and line_end_time_ms > words[i]["start"]:
                         line_end_time_ms = words[i]["start"] - 10 

                srt_entries.append(f"{srt_idx}\n{format_time_srt(current_line_start_time_ms / 1000.0)} --> {format_time_srt(line_end_time_ms / 1000.0)}\n{current_line_text.strip()}")
                srt_idx += 1
                
                current_line_text = word_text
                current_line_start_time_ms = word_start_ms
            else:
                current_line_text = potential_line

    if current_line_text and current_line_start_time_ms != -1:
        line_end_time_ms = words[-1]["end"]
        line_duration_s = (line_end_time_ms - current_line_start_time_ms) / 1000.0
        if line_duration_s < min_line_duration_s:
            line_end_time_ms = current_line_start_time_ms + int(min_line_duration_s * 1000)
        
        srt_entries.append(f"{srt_idx}\n{format_time_srt(current_line_start_time_ms / 1000.0)} --> {format_time_srt(line_end_time_ms / 1000.0)}\n{current_line_text.strip()}")

    return "\n\n".join(srt_entries)


async def _format_claude_subtitles_to_srt(claude_subtitles_data: List[Dict], scenes_data: List[Dict], total_video_duration: float, request_id: str) -> str:
    srt_entries = []
    entry_counter = 1
    current_timeline_pos = 0.0 

    if not claude_subtitles_data: return ""

    scene_start_times = {}
    cumulative_time = 0.0
    for i, scene in enumerate(scenes_data):
        scene_start_times[i] = cumulative_time
        duration = scene.get('final_duration') 
        if duration is None:
            duration = scene.get("duration_seconds", 5.0) 
        try: 
            duration = float(duration)
        except (ValueError, TypeError): 
            logger.warning(f"[{request_id}] Invalid duration for scene {i} ('{scene.get('duration_seconds')}') for subtitle timing. Using default 5s.")
            duration = 5.0
        cumulative_time += duration
    
    effective_total_duration = total_video_duration if total_video_duration > 0 else cumulative_time or (len(claude_subtitles_data) * 3.0)

    for i, sub_entry in enumerate(claude_subtitles_data):
        text = sub_entry.get("text")
        if not text: continue

        start_time_sec = current_timeline_pos 
        estimated_duration = max(2.0, min(len(text.split()) * 0.4, 7.0)) 

        timing_instr = sub_entry.get("timing_instructions", "").lower()
        scene_match = re.search(r"scene\s*(\d+)", timing_instr)
        offset_match = re.search(r"(\d+)\s*seconds\s*into\s*scene\s*(\d+)", timing_instr)

        if offset_match:
            try:
                offset_val = int(offset_match.group(1)); scene_num_one_based = int(offset_match.group(2)); scene_idx = scene_num_one_based - 1
                if 0 <= scene_idx < len(scenes_data): start_time_sec = scene_start_times.get(scene_idx, current_timeline_pos) + offset_val
            except ValueError: logger.warning(f"[{request_id}] Invalid offset/scene number in timing: {timing_instr}")
        elif scene_match:
            try:
                scene_idx = int(scene_match.group(1)) - 1
                if 0 <= scene_idx < len(scenes_data): start_time_sec = scene_start_times.get(scene_idx, current_timeline_pos)
            except ValueError: logger.warning(f"[{request_id}] Invalid scene number in timing: {timing_instr}")
        
        end_time_sec = start_time_sec + estimated_duration
        end_time_sec = min(end_time_sec, effective_total_duration)
        
        if i + 1 < len(claude_subtitles_data):
            next_sub_timing_instr = claude_subtitles_data[i+1].get("timing_instructions", "").lower()
            next_scene_match_next = re.search(r"scene\s*(\d+)", next_sub_timing_instr)
            next_offset_match_next = re.search(r"(\d+)\s*seconds\s*into\s*scene\s*(\d+)", next_sub_timing_instr)
            next_start_time_candidate = effective_total_duration 
            if next_offset_match_next:
                try:
                    next_offset_val = int(next_offset_match_next.group(1)); next_scene_num_one_based = int(next_offset_match_next.group(2)); next_scene_idx = next_scene_num_one_based -1
                    if 0 <= next_scene_idx < len(scenes_data): next_start_time_candidate = scene_start_times.get(next_scene_idx, cumulative_time) + next_offset_val 
                except ValueError: pass
            elif next_scene_match_next:
                try:
                    next_scene_idx = int(next_scene_match_next.group(1)) - 1
                    if 0 <= next_scene_idx < len(scenes_data): next_start_time_candidate = scene_start_times.get(next_scene_idx, cumulative_time) 
                except ValueError: pass
            if end_time_sec > next_start_time_candidate: end_time_sec = next_start_time_candidate

        if start_time_sec >= end_time_sec : end_time_sec = start_time_sec + 0.5 
        srt_entries.append(f"{entry_counter}\n{format_time_srt(start_time_sec)} --> {format_time_srt(end_time_sec)}\n{text}")
        entry_counter += 1; current_timeline_pos = end_time_sec 
    return "\n\n".join(srt_entries)

async def _translate_srt_content(srt_content: str, target_lang: str, source_lang: Optional[str], request: Request, request_id: str, generated_files_summary_ref: dict) -> str:
    if not hasattr(request.app.state, 'deepl_translator') or request.app.state.deepl_translator is None:
        logger.warning(f"[{request_id}] DeepL translator not available in app.state, skipping translation.")
        return srt_content

    deepl_translator = request.app.state.deepl_translator
    
    translated_lines = []
    srt_blocks = srt_content.strip().split('\n\n')
    texts_to_translate = []
    original_blocks_info = []

    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3 and "-->" in lines[1]:
            text_content = "\n".join(lines[2:])
            texts_to_translate.append(text_content)
            original_blocks_info.append({"header": f"{lines[0]}\n{lines[1]}", "text_line_count": len(lines[2:])})
        else:
            original_blocks_info.append({"header": block, "text_line_count": 0})

    if not texts_to_translate: 
        logger.info(f"[{request_id}] No text found in SRT to translate.")
        return srt_content

    logger.info(f"[{request_id}] Translating {len(texts_to_translate)} subtitle segments to {target_lang} from {source_lang or 'auto'}")
    
    try:
        from config import DEEPL_LANG_MAP
        api_target_lang = DEEPL_LANG_MAP.get(target_lang.lower())
        api_source_lang = DEEPL_LANG_MAP.get(source_lang.lower()) if source_lang else None
        if not api_target_lang:
            logger.error(f"[{request_id}] DeepL target language code for '{target_lang}' not found."); generated_files_summary_ref["errors"].append({"step": "subtitle_translation", "error": f"DeepL unsupported target language: {target_lang}"}); return srt_content
        
        translated_results = await asyncio.to_thread(
            deepl_translator.translate_text,
            texts_to_translate,
            target_lang=api_target_lang,
            source_lang=api_source_lang
        )
        
        translated_texts_iter = iter([res.text for res in translated_results])
        reconstructed_srt_blocks = []
        
        for block_info in original_blocks_info:
            if block_info["text_line_count"] > 0 : 
                try:
                    translated_text = next(translated_texts_iter)
                    reconstructed_srt_blocks.append(f"{block_info['header']}\n{translated_text}")
                except StopIteration: 
                     logger.error(f"[{request_id}] Mismatch between original and translated text lines for SRT reconstruction. Original text lines: {len(texts_to_translate)}, Processed translations: {len(reconstructed_srt_blocks)}") 
                     reconstructed_srt_blocks.append(block_info['header'] + "\n[Translation Error - Mismatch]")
            else:
                reconstructed_srt_blocks.append(block_info["header"])
        
        logger.info(f"[{request_id}] Subtitle translation successful.")
        return "\n\n".join(reconstructed_srt_blocks)
        
    except Exception as e:
        logger.error(f"[{request_id}] DeepL translation error: {e}", exc_info=True)
        generated_files_summary_ref["errors"].append({"step": "subtitle_translation", "error": f"DeepL translation failed: {str(e)}"})
        return srt_content


async def create_video_from_text_pipeline(req: VideoGenerationRequest, fastapi_request: Request, task_id: str, original_task_details: Optional[Dict] = None, edit_request_params: Optional[VideoEditRequest] = None):
    request_id = task_id 
    temp_dir = TEMP_DIR / request_id 
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files_summary = {
        "request_id": request_id,
        "params": req.model_dump(),
        "claude_analysis": None,
        "scenes": [], 
        "narration_audios": [], 
        "subtitle_file": None, 
        "bgm_audio_file": None, 
        "final_video_path": None,
        "final_video_url": None,
        "errors": [],
        "ffmpeg_video_concat_log": None,
        "temp_concat_video_path": None,
        "ffmpeg_final_video_log": None,
        "summary_file_path": str(temp_dir / f"generation_summary_{request_id}.json")
    }

    is_editing = original_task_details is not None and edit_request_params is not None
    
    current_step = "initializing"
    parsed_prompt_data = {}
    
    if is_editing and original_task_details.get("claude_analysis"):
        # Start with a copy of the original Claude analysis
        parsed_prompt_data = json.loads(json.dumps(original_task_details["claude_analysis"])) # Deep copy
        generated_files_summary["claude_analysis"] = parsed_prompt_data
        logger.info(f"[{request_id}] Loaded Claude analysis from original task.")

    actual_total_processed_video_duration = 0.0
    final_video_successfully_moved = False

    try:
        save_status(task_id, {"status": "processing", "message": "Video generation process initiated.", "current_step": current_step, "details": generated_files_summary})

        # 1. Claude Prompt Analysis (or modification if editing)
        current_step = "claude_analysis"
        needs_claude_reanalysis = not parsed_prompt_data
        
        if is_editing:
            # If overall prompt or scene prompts change, or full narration script changes, then re-analyze.
            if req.prompt != original_task_details["params"]["prompt"] or \
               req.scene_prompts != original_task_details["params"].get("scene_prompts") or \
               (edit_request_params.edit_target == EditTargetEnum.NARRATION_FULL_SCRIPT and edit_request_params.new_prompt):
                needs_claude_reanalysis = True
                logger.info(f"[{request_id}] Claude re-analysis triggered by edit of main prompt, scene prompts, or full narration script.")
        
        if needs_claude_reanalysis:
            logger.info(f"[{request_id}] Step 1: Running Claude analysis...")
            claude_system_prompt = f"""
You are an AI video production planner. Based on the user's request, create a detailed plan in JSON format.
The JSON must include: "scenes", "narration", "subtitles", "bgm".

User's overall video theme: {req.prompt}
Target video duration: {req.duration} seconds. Resolution: {req.resolution}.
Narration: {'Enabled' if req.narration_enabled else 'Disabled'}, Language: {req.narration_lang}.
Subtitles: {'Enabled' if req.subtitles_enabled else 'Disabled'}.
BGM: {'Enabled' if req.bgm_enabled else 'Disabled'}

"scenes": List of objects. Each must have "scene_description" (detailed prompt for image/video gen) and "duration_seconds" (integer, e.g., 5). The sum of these durations should ideally match the user's target video duration.
If user provided scene_prompts ({req.scene_prompts}), use them. Otherwise, break down the main prompt.
Assign "narration_segment_indices" and "subtitle_indices" if applicable.

"narration": Object. If narration is enabled:
"full_script": (String) Complete script.
"segments": List of objects, each with "text" (string) and "voice_instructions" (string, e.g., "energetic").
Base this on narration_script_prompt ({req.narration_script_prompt}) or the main theme.

"subtitles": (List of Objects, Required if subtitles are enabled) A list of subtitle objects, where each object must contain:
    *   "text": (String, Required) The text to be displayed as a subtitle.
    *   "timing_instructions": (String, Optional) User-provided cues for when this subtitle should appear/disappear (e.g., "scene 1, 0s-5s", "scene 2, starts 2s in, lasts 3s"). If not provided, timing will be inferred.

"bgm": An object containing background music details:
    *   "description": (String, Required if BGM is enabled) A detailed description of the desired background music (e.g., "upbeat electronic, 120 BPM", "calm Lo-fi hip hop").

Ensure total scene duration roughly matches target video duration. Output valid JSON.
"""
            claude_user_prompt_parts = [f"Overall video theme: {req.prompt}"]
            if req.scene_prompts: claude_user_prompt_parts.append(f"Scene-specific prompts: {'; '.join(req.scene_prompts)}")
            if req.narration_script_prompt: claude_user_prompt_parts.append(f"Narration instructions: {req.narration_script_prompt}")
            if req.subtitle_script_prompt and not (is_editing and edit_request_params and edit_request_params.edit_target == EditTargetEnum.SUBTITLE_FULL_SCRIPT):
                # Only include subtitle_script_prompt if not directly providing an SRT
                claude_user_prompt_parts.append(f"Subtitle instructions: {req.subtitle_script_prompt}")
            if req.bgm_prompt: claude_user_prompt_parts.append(f"BGM instructions: {req.bgm_prompt}")
            claude_user_prompt = "\n".join(claude_user_prompt_parts) + "\n\nPlease generate a video plan based on these inputs."
            
            save_status(task_id, {"status": "processing", "message": "Analyzing prompt with Claude...", "current_step": current_step, "details": generated_files_summary})
            claude_response_obj = await get_claude_response(fastapi_request, claude_user_prompt, claude_system_prompt, "claude-3-opus-20240229")
            if claude_response_obj.error or not claude_response_obj.response:
                err_msg = f"Claude API call failed: {claude_response_obj.error or 'No response'}"
                generated_files_summary["errors"].append({"step": current_step, "error": err_msg})
                raise ValueError(err_msg) 
            
            try:
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', claude_response_obj.response, re.DOTALL)
                if not json_match: json_match = re.search(r'(\{[\s\S]*?\})', claude_response_obj.response, re.DOTALL)
                if not json_match: raise ValueError("No JSON object found in Claude's response.")
                json_str = json_match.group(1)
                parsed_prompt_data = json.loads(json_str)
                generated_files_summary["claude_analysis"] = parsed_prompt_data
                logger.info(f"[{request_id}] Claude analysis successful.")
            except (json.JSONDecodeError, ValueError) as e:
                err_msg = f"Failed to parse JSON from Claude: {e}. Response: {claude_response_obj.response[:500]}"
                generated_files_summary["errors"].append({"step": current_step, "error": err_msg})
                raise ValueError(err_msg)
        else: # is_editing and no full re-analysis needed
            logger.info(f"[{request_id}] Using existing Claude data, potentially modified by edit request.")
            # Apply specific edits to the existing parsed_prompt_data
            if edit_request_params.edit_target == EditTargetEnum.SCENE_DURATION and edit_request_params.target_indices and edit_request_params.new_value_numeric is not None:
                for idx in edit_request_params.target_indices:
                    if 0 <= idx < len(parsed_prompt_data.get("scenes", [])):
                        parsed_prompt_data["scenes"][idx]["duration_seconds"] = edit_request_params.new_value_numeric
                        logger.info(f"[{request_id}] Updated duration for scene {idx} to {edit_request_params.new_value_numeric}s in existing plan.")
            elif edit_request_params.edit_target == EditTargetEnum.NARRATION_SEGMENT_TEXT and edit_request_params.target_indices and edit_request_params.new_prompt:
                for idx in edit_request_params.target_indices:
                    if 0 <= idx < len(parsed_prompt_data.get("narration", {}).get("segments", [])):
                        parsed_prompt_data["narration"]["segments"][idx]["text"] = edit_request_params.new_prompt
                        logger.info(f"[{request_id}] Updated text for narration segment {idx} in existing plan.")
            elif edit_request_params.edit_target == EditTargetEnum.BGM and edit_request_params.new_prompt:
                if "bgm" not in parsed_prompt_data: parsed_prompt_data["bgm"] = {}
                parsed_prompt_data["bgm"]["description"] = edit_request_params.new_prompt
                logger.info(f"[{request_id}] Updated BGM prompt in existing plan.")
            generated_files_summary["claude_analysis"] = parsed_prompt_data # Save potentially modified plan

        save_status(task_id, {"status": "processing", "message": "Claude analysis/update complete.", "current_step": current_step, "details": generated_files_summary})

        # 2. Video Material Generation
        current_step = "video_material_generation"
        logger.info(f"[{request_id}] Step 2: Generating/Reusing video/image materials...")
        scenes_plan = parsed_prompt_data.get("scenes", [])
        if not scenes_plan:
            scenes_plan = [{"scene_description": req.prompt, "duration_seconds": req.duration or 10}]
            logger.warning(f"[{request_id}] No scenes in plan. Using main prompt for one scene.")
            if "scenes" not in parsed_prompt_data: parsed_prompt_data["scenes"] = [] # Initialize if not present
            parsed_prompt_data["scenes"] = scenes_plan
            generated_files_summary["claude_analysis"]["scenes"] = scenes_plan


        target_w_req, target_h_req = parse_resolution(req.resolution)
        video_input_options: List[str] = [] 
        video_filter_inputs: List[str] = []
        actual_total_processed_video_duration = 0.0
        
        num_scenes = len(scenes_plan)
        default_scene_duration = (req.duration / num_scenes) if num_scenes > 0 and req.duration else 5.0
        default_scene_duration = max(1.0, default_scene_duration)

        replicate_model_to_use = req.replicate_model_version or REPLICATE_ZEROSCOPE_MODEL_XL
        logger.info(f"[{request_id}] Using Replicate model version: {replicate_model_to_use}")

        new_scenes_data = [] 

        async with httpx.AsyncClient() as client:
            for i, scene_info_claude in enumerate(scenes_plan):
                scene_summary_entry = {
                    "scene_index": i,
                    "description_from_claude": scene_info_claude.get("scene_description"),
                    "duration_seconds_claude": scene_info_claude.get("duration_seconds"), # Original duration from Claude
                    "path": None, "source_api": None, "error": None, "material_type": None,
                    "final_duration": None # Actual duration used for this scene
                }

                scene_desc = scene_info_claude.get("scene_description", req.prompt)
                try:
                    scene_duration_seconds = float(scene_info_claude.get("duration_seconds", default_scene_duration))
                    if scene_duration_seconds <= 0: scene_duration_seconds = 1.0
                except (ValueError, TypeError):
                    scene_duration_seconds = default_scene_duration
                scene_summary_entry["final_duration"] = scene_duration_seconds # This might be overridden by edit_request

                should_regenerate_scene = not is_editing # Default to true for new generations

                if is_editing and edit_request_params:
                    if edit_request_params.edit_target == EditTargetEnum.SCENE_VIDEO and req.target_indices and i in edit_request_params.target_indices:
                        scene_desc = edit_request_params.new_prompt or scene_desc
                        scene_summary_entry["description_from_claude"] = scene_desc # Update description for regeneration
                        logger.info(f"[{request_id}] Scene {i} will be regenerated with new prompt: {scene_desc[:50]}...")
                        should_regenerate_scene = True
                    elif edit_request_params.edit_target == EditTargetEnum.SCENE_DURATION and req.target_indices and i in edit_request_params.target_indices and edit_request_params.new_value_numeric is not None:
                        scene_duration_seconds = float(edit_request_params.new_value_numeric)
                        scene_summary_entry["final_duration"] = scene_duration_seconds
                        logger.info(f"[{request_id}] Scene {i} duration updated to {scene_duration_seconds}s. Will regenerate if needed.")
                        should_regenerate_scene = True # Assume regeneration for simplicity, especially if it's a video
                    elif original_task_details and original_task_details.get("scenes") and i < len(original_task_details["scenes"]):
                        original_scene_data = original_task_details["scenes"][i]
                        original_file_path_str = original_task_details.get("final_video_path") # Base path for original assets
                        if original_file_path_str and original_scene_data.get("path"):
                            # Construct the full path to the original asset
                            original_asset_path = Path(original_scene_data["path"]) # This should be the path within the original task's temp or static dir
                            
                            # If the original path is relative to its temp dir, we need to adjust it
                            if not original_asset_path.is_absolute():
                                original_asset_path = Path(original_task_details.get("summary_file_path", "")).parent / original_asset_path.name

                            if original_asset_path.exists():
                                new_asset_path = temp_dir / original_asset_path.name
                                try:
                                    shutil.copy(original_asset_path, new_asset_path)
                                    scene_summary_entry.update(original_scene_data) # Copy all info
                                    scene_summary_entry["path"] = str(new_asset_path.resolve()) # Update path
                                    scene_summary_entry["final_duration"] = scene_duration_seconds # Update duration if changed
                                    should_regenerate_scene = False
                                    logger.info(f"[{request_id}] Reusing material for scene {i}: {new_asset_path}")
                                except Exception as e_copy:
                                    logger.warning(f"[{request_id}] Could not copy original asset for scene {i} from {original_asset_path}: {e_copy}. Will regenerate.")
                                    should_regenerate_scene = True
                            else:
                                logger.warning(f"[{request_id}] Original asset for scene {i} not found at {original_asset_path}, will regenerate.")
                                should_regenerate_scene = True
                        else:
                            should_regenerate_scene = True 
                    else:
                        should_regenerate_scene = True # Fallback to regenerate

                if should_regenerate_scene:
                    scene_summary_entry["path"] = None # Clear any old path if regenerating
                    final_replicate_prompt_for_api = f"{req.replicate_prompt_prefix}, {scene_desc}" if req.replicate_prompt_prefix else scene_desc
                    
                    if REPLICATE_API_TOKEN:
                        logger.info(f"[{request_id}] Scene {i+1}: Generating with Replicate for '{final_replicate_prompt_for_api[:50]}...'")
                        replicate_width, replicate_height = (576, 320) if replicate_model_to_use == REPLICATE_ZEROSCOPE_MODEL_576W else (1024, 576)
                        
                        replicate_payload_input = {
                            "prompt": final_replicate_prompt_for_api,
                            "negative_prompt": req.replicate_negative_prompt,
                            "width": replicate_width,
                            "height": replicate_height,
                            "num_frames": max(16, min(int(scene_summary_entry["final_duration"] * (req.fps or DEFAULT_FPS)), 60)),
                            "fps": req.fps or DEFAULT_FPS,
                            "guidance_scale": req.replicate_guidance_scale,
                            "num_inference_steps": req.replicate_num_inference_steps,
                        }
                        if req.replicate_seed is not None and req.replicate_seed != -1:
                            replicate_payload_input["seed"] = req.replicate_seed
                        
                        logger.info(f"[{request_id}] Replicate API Input for scene {i+1}: {replicate_payload_input}")
                        rep_payload = {"version": replicate_model_to_use, "input": replicate_payload_input}
                            
                        try:
                            prediction_response = await client.post(REPLICATE_API_URL, headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"}, json=rep_payload, timeout=REPLICATE_API_TIMEOUT)
                            prediction_response.raise_for_status(); pred_data = prediction_response.json()
                            prediction_id = pred_data.get("id"); get_url = pred_data.get("urls", {}).get("get")

                            if prediction_id and get_url:
                                for _ in range(REPLICATE_MAX_POLL_ATTEMPTS):
                                    await asyncio.sleep(REPLICATE_POLL_INTERVAL)
                                    poll_response = await client.get(get_url, headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"})
                                    poll_response.raise_for_status(); result_data = poll_response.json()
                                    if result_data.get("status") == "succeeded":
                                        output_url = result_data.get("output"); output_url = output_url[0] if isinstance(output_url, list) and output_url else None
                                        if output_url:
                                            video_file_response = await client.get(output_url, timeout=REPLICATE_API_TIMEOUT); video_file_response.raise_for_status()
                                            scene_file_path = temp_dir / f"scene_{i}_replicate.mp4"; scene_file_path.write_bytes(video_file_response.content)
                                            scene_summary_entry["path"] = str(scene_file_path.resolve()); scene_summary_entry["source_api"] = "replicate"
                                            scene_summary_entry["material_type"] = "video"
                                            logger.info(f"[{request_id}] Scene {i+1} Replicate success: {scene_file_path.name}")
                                        else: scene_summary_entry["error"] = "Replicate succeeded but no output URL."
                                        break 
                                    elif result_data.get("status") == "failed": scene_summary_entry["error"] = f"Replicate prediction failed: {result_data.get('error', 'Unknown error')}"; break
                                else: scene_summary_entry["error"] = "Replicate prediction polling timed out."
                            else: scene_summary_entry["error"] = "Failed to start Replicate prediction."
                        except Exception as e_replicate:
                            logger.warning(f"[{request_id}] Scene {i+1} Replicate API error: {e_replicate}", exc_info=True)
                            scene_summary_entry["error"] = (scene_summary_entry.get("error","") + f"; Replicate API error: {str(e_replicate)[:100]}").strip("; ")
                    
                    if not scene_summary_entry["path"] and STABILITY_API_KEY:
                        logger.info(f"[{request_id}] Scene {i+1}: Fallback to Stability Text-to-Image for '{scene_desc[:30]}...'")
                        try:
                            img_width, img_height = target_w_req, target_h_req
                            max_dim = 1024 
                            if img_width > max_dim or img_height > max_dim:
                                if img_width > img_height: img_height = int(max_dim * img_height / img_width); img_width = max_dim
                                else: img_width = int(max_dim * img_width / img_height); img_height = max_dim
                            img_width = (img_width // 64) * 64; img_height = (img_height // 64) * 64
                            if img_width == 0: img_width = 512
                            if img_height == 0: img_height = 512
                            
                            stability_api_url_txt2img = STABILITY_TEXT_TO_IMAGE_API_URL_BASE.format(engine_id=STABILITY_DEFAULT_ENGINE)
                            stab_payload = {"text_prompts": [{"text": f"{scene_desc}, cinematic, {req.video_quality}"}], "height": img_height, "width": img_width, "samples": 1, "steps": 30 }
                            stability_response = await client.post(stability_api_url_txt2img, headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}, json=stab_payload, timeout=STABILITY_API_TIMEOUT)
                            stability_response.raise_for_status(); image_data_json = stability_response.json()
                            if image_data_json.get("artifacts"):
                                base64_image = image_data_json["artifacts"][0].get("base64")
                                if base64_image:
                                    image_bytes = base64.b64decode(base64_image)
                                    scene_file_path_stab = temp_dir / f"scene_{i}_stability.png"; scene_file_path_stab.write_bytes(image_bytes)
                                    scene_summary_entry["path"] = str(scene_file_path_stab.resolve()); scene_summary_entry["source_api"] = "stability_text_to_image"
                                    scene_summary_entry["material_type"] = "image"
                                    logger.info(f"[{request_id}] Scene {i+1} Stability image success: {scene_file_path_stab.name}")
                                else: scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability: No image data.").strip("; ")
                            else: scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability: No artifacts.").strip("; ")
                        except Exception as e_stability:
                            logger.warning(f"[{request_id}] Scene {i+1} Stability API error: {e_stability}", exc_info=True)
                            scene_summary_entry["error"] = (scene_summary_entry.get("error","") + f"; Stability API error: {str(e_stability)[:100]}").strip("; ")
                    elif not STABILITY_API_KEY and not scene_summary_entry["path"]:
                         logger.info(f"[{request_id}] Scene {i+1}: STABILITY_API_KEY not set. Skipping Stability AI.")
                         scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability API key not configured.").strip("; ") if scene_summary_entry.get("error") else "Stability API key not configured."

                material_path_str = scene_summary_entry.get("path")
                if material_path_str and Path(material_path_str).is_file():
                    material_path_obj = Path(material_path_str)
                    
                    if scene_summary_entry["material_type"] == "image":
                        video_input_options.extend(["-loop", "1", "-r", str(DEFAULT_FPS), "-t", str(scene_summary_entry["final_duration"]), "-i", str(material_path_obj.resolve())])
                    else: # video file
                        video_input_options.extend(["-i", str(material_path_obj.resolve())])
                    
                    video_filter_inputs.append(f"[{len(new_scenes_data)}:v]") # Index based on current position in new_scenes_data
                    actual_total_processed_video_duration += float(scene_summary_entry["final_duration"])
                elif not scene_summary_entry["error"]:
                     scene_summary_entry["error"] = "All material generation attempts failed or were skipped for this scene."
                
                new_scenes_data.append(scene_summary_entry)
                if scene_summary_entry.get("error"): 
                    logger.error(f"[{request_id}] Scene {i+1} final error: {scene_summary_entry['error']}")
                    generated_files_summary["errors"].append({"step": current_step, "scene_index": i, "error": scene_summary_entry['error']})
                save_status(task_id, {"status": "processing", "message": f"Processed material for scene {i+1}/{len(scenes_plan)}", "current_step": current_step, "details": generated_files_summary})
                await asyncio.sleep(0.1) 
        
        generated_files_summary["scenes"] = new_scenes_data
        logger.info(f"[{request_id}] --- Finished Video Material Generation ---")
        logger.info(f"[{request_id}] Video input options: {video_input_options}")
        logger.info(f"[{request_id}] Video filter inputs: {video_filter_inputs}")
        logger.info(f"[{request_id}] Actual total processed video duration: {actual_total_processed_video_duration} seconds")

        # 3. Narration Generation
        current_step = "narration_generation"
        logger.info(f"[{request_id}] Step 3: Generating/Reusing narration...")
        if req.narration_enabled and ELEVENLABS_API_KEY:
            narration_plan = parsed_prompt_data.get("narration", {})
            narration_segments_claude = narration_plan.get("segments", []) # These might have been updated by edit logic
            
            if narration_segments_claude:
                regenerated_narration_audios = []
                async with httpx.AsyncClient(timeout=ELEVENLABS_API_TIMEOUT) as client_el:
                    for i_nar, seg_info_el in enumerate(narration_segments_claude):
                        nar_summary_entry = {"segment_index": i_nar, "text": seg_info_el.get("text"), "path": None, "error": None, "selected_voice_id": None}
                        segment_text_to_speak = seg_info_el.get("text")
                        voice_id_to_use = req.narration_voice_id or get_default_voice_id(req.narration_lang)

                        should_regenerate_narration = not is_editing # Always generate for new tasks
                        if is_editing and edit_request_params:
                            if edit_request_params.edit_target == EditTargetEnum.NARRATION_SEGMENT_TEXT and edit_request_params.target_indices and i_nar in edit_request_params.target_indices:
                                # Text already updated in parsed_prompt_data if applicable
                                logger.info(f"[{request_id}] Regenerating narration segment {i_nar} due to text change.")
                                should_regenerate_narration = True
                            elif edit_request_params.edit_target == EditTargetEnum.NARRATION_SEGMENT_VOICE and edit_request_params.target_indices and i_nar in edit_request_params.target_indices and edit_request_params.new_value_string:
                                voice_id_to_use = edit_request_params.new_value_string
                                logger.info(f"[{request_id}] Regenerating narration segment {i_nar} with new voice: {voice_id_to_use}.")
                                should_regenerate_narration = True
                            elif edit_request_params.edit_target == EditTargetEnum.NARRATION_FULL_SCRIPT:
                                should_regenerate_narration = True
                            elif original_task_details and original_task_details.get("narration_audios") and i_nar < len(original_task_details["narration_audios"]):
                                original_nar_audio = original_task_details["narration_audios"][i_nar]
                                if original_nar_audio.get("path") and Path(original_nar_audio["path"]).exists() and \
                                   original_nar_audio.get("text") == segment_text_to_speak and \
                                   original_nar_audio.get("selected_voice_id") == (req.narration_voice_id or get_default_voice_id(req.narration_lang)): # Compare with current global or default voice
                                    original_audio_path = Path(original_nar_audio["path"])
                                    new_audio_path = temp_dir / original_audio_path.name
                                    if original_audio_path != new_audio_path:
                                        shutil.copy(original_audio_path, new_audio_path)
                                    nar_summary_entry.update(original_nar_audio)
                                    nar_summary_entry["path"] = str(new_audio_path.resolve())
                                    should_regenerate_narration = False
                                    logger.info(f"[{request_id}] Reusing narration segment {i_nar}: {new_audio_path}")
                                else:
                                    should_regenerate_narration = True # Text or voice changed, or file missing
                            else:
                                should_regenerate_narration = True
                        
                        if not segment_text_to_speak or not segment_text_to_speak.strip():
                            nar_summary_entry["error"] = "Empty text."
                            regenerated_narration_audios.append(nar_summary_entry)
                            continue
                        
                        nar_summary_entry["selected_voice_id"] = voice_id_to_use

                        if should_regenerate_narration:
                            el_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id_to_use}"
                            payload_el = {"text": segment_text_to_speak, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
                            try:
                                response_tts = await client_el.post(el_url, headers={"xi-api-key": ELEVENLABS_API_KEY, "Accept": "audio/mpeg"}, json=payload_el)
                                response_tts.raise_for_status()
                                narration_file_p = temp_dir / f"narration_segment_{i_nar}.mp3"; narration_file_p.write_bytes(response_tts.content)
                                nar_summary_entry["path"] = str(narration_file_p.resolve())
                                logger.info(f"[{request_id}] Narration segment {i_nar+1} generated: {narration_file_p.name}")
                            except Exception as e_elevenlabs: 
                                logger.error(f"[{request_id}] ElevenLabs API error for segment {i_nar+1}: {e_elevenlabs}", exc_info=True)
                                nar_summary_entry["error"] = f"ElevenLabs API error: {str(e_elevenlabs)[:100]}"
                        
                        regenerated_narration_audios.append(nar_summary_entry)
                        if nar_summary_entry.get("error"): generated_files_summary["errors"].append({"step": current_step, "segment_index": i_nar, "error": nar_summary_entry['error']})
                        save_status(task_id, {"status": "processing", "message": f"Processed narration for segment {i_nar+1}/{len(narration_segments_claude)}", "current_step": current_step, "details": generated_files_summary})
                        await asyncio.sleep(0.5) 
                generated_files_summary["narration_audios"] = regenerated_narration_audios
            else: logger.info(f"[{request_id}] No narration segments from Claude.")
        elif not req.narration_enabled: logger.info(f"[{request_id}] Narration disabled.")
        else: 
            logger.warning(f"[{request_id}] Narration enabled but ELEVENLABS_API_KEY missing.")
            generated_files_summary["errors"].append({"step": current_step, "error": "ELEVENLABS_API_KEY missing for narration."})
        save_status(task_id, {"status": "processing", "message": "Narration generation step finished.", "current_step": current_step, "details": generated_files_summary})
        
        # 4. Subtitle Generation
        current_step = "subtitle_generation"
        logger.info(f"[{request_id}] Step 4: Generating/Reusing subtitles...")
        srt_content_str: Optional[str] = None
        subtitle_file_path_str = None # Will hold the path to the final SRT file for this task

        if req.subtitles_enabled:
            should_regenerate_srt = True # Default to regeneration

            if is_editing and edit_request_params:
                if edit_request_params.edit_target == EditTargetEnum.SUBTITLE_FULL_SCRIPT and edit_request_params.new_script_srt is not None:
                    srt_content_str = edit_request_params.new_script_srt
                    logger.info(f"[{request_id}] Using new user-provided SRT script from edit_request.")
                    should_regenerate_srt = False # Content is directly provided
                elif edit_request_params.edit_target == EditTargetEnum.SUBTITLE_STYLE:
                    # If only style changes, we need the content. Try to reuse.
                    if original_task_details and original_task_details.get("subtitle_file"):
                        original_srt_path = Path(original_task_details["subtitle_file"])
                        if original_srt_path.exists():
                            new_srt_path = temp_dir / original_srt_path.name
                            shutil.copy(original_srt_path, new_srt_path)
                            generated_files_summary["subtitle_file"] = str(new_srt_path.resolve())
                            with open(new_srt_path, "r", encoding="utf-8") as f:
                                srt_content_str = f.read()
                            logger.info(f"[{request_id}] Reusing SRT content from {new_srt_path} for style change.")
                            should_regenerate_srt = False # Content reused, only styling will be applied by ffmpeg
                        else:
                             logger.warning(f"[{request_id}] Original subtitle file {original_srt_path} not found for style edit. Will attempt to regenerate.")
                             should_regenerate_srt = True
                    else:
                        logger.warning(f"[{request_id}] Subtitle style edit requested, but no original subtitle file info. Will attempt to regenerate.")
                        should_regenerate_srt = True
                elif edit_request_params.edit_target in [EditTargetEnum.SCENE_VIDEO, EditTargetEnum.SCENE_DURATION, EditTargetEnum.NARRATION_SEGMENT_TEXT, EditTargetEnum.NARRATION_FULL_SCRIPT, EditTargetEnum.NARRATION_SEGMENT_VOICE]:
                    # If video or narration changes, subtitles likely need regeneration
                    logger.info(f"[{request_id}] Video or narration changed, subtitles will be regenerated.")
                    should_regenerate_srt = True
                elif original_task_details and original_task_details.get("subtitle_file") and Path(original_task_details["subtitle_file"]).exists():
                    # Default for other edits: try to reuse if no specific subtitle edit
                    original_srt_path = Path(original_task_details["subtitle_file"])
                    new_srt_path = temp_dir / original_srt_path.name
                    shutil.copy(original_srt_path, new_srt_path)
                    generated_files_summary["subtitle_file"] = str(new_srt_path.resolve())
                    with open(new_srt_path, "r", encoding="utf-8") as f:
                        srt_content_str = f.read()
                    logger.info(f"[{request_id}] Reusing existing SRT file: {new_srt_path}")
                    should_regenerate_srt = False


            if should_regenerate_srt:
                valid_narration_paths_for_srt = [
                    nar_info["path"] 
                    for nar_info in generated_files_summary.get("narration_audios", []) 
                    if nar_info.get("path") and Path(nar_info["path"]).exists()
                ]
                
                if ASSEMBLYAI_API_KEY and valid_narration_paths_for_srt:
                    logger.info(f"[{request_id}] Attempting subtitle generation via AssemblyAI from narration.")
                    audio_for_transcription_path_str = None
                    concatenated_narration_path_for_srt = None 
                    if len(valid_narration_paths_for_srt) > 1:
                        narration_concat_list_file = temp_dir / f"narrations_for_srt_concat_{request_id}.txt"
                        with open(narration_concat_list_file, "w") as f_concat:
                            for p in valid_narration_paths_for_srt:
                                f_concat.write(f"file '{Path(p).resolve().as_posix()}'\n")
                        
                        concatenated_narration_path_for_srt = temp_dir / f"concatenated_narration_for_srt_{request_id}.mp3"
                        concat_cmd_srt = [FFMPEG_COMMAND, "-y", "-f", "concat", "-safe", "0", "-i", str(narration_concat_list_file), "-c", "copy", str(concatenated_narration_path_for_srt)]
                        concat_success_srt, _, concat_stderr_srt = await run_ffmpeg_command_async(concat_cmd_srt, request_id + "_narration_srt_concat")
                        if concat_success_srt:
                            audio_for_transcription_path_str = str(concatenated_narration_path_for_srt.resolve())
                        else:
                            logger.error(f"[{request_id}] Failed to concatenate narration files for SRT: {concat_stderr_srt}")
                            generated_files_summary["errors"].append({"step": current_step, "error": f"Failed to concatenate narration files for SRT: {concat_stderr_srt}"})
                    elif len(valid_narration_paths_for_srt) == 1:
                        audio_for_transcription_path_str = valid_narration_paths_for_srt[0]
                    
                    if audio_for_transcription_path_str:
                        logger.info(f"[{request_id}] Transcribing audio file for subtitles: {audio_for_transcription_path_str}")
                        async with httpx.AsyncClient(timeout=ASSEMBLYAI_API_TIMEOUT) as client_asm:
                            try:
                                with open(audio_for_transcription_path_str, "rb") as f_audio:
                                    files = {'file': (Path(audio_for_transcription_path_str).name, f_audio, 'audio/mpeg')}
                                    upload_response = await client_asm.post("https://api.assemblyai.com/v2/upload", headers={"authorization": ASSEMBLYAI_API_KEY}, files=files)
                                upload_response.raise_for_status()
                                upload_url_assembly = upload_response.json().get("upload_url")

                                if not upload_url_assembly: raise Exception("AssemblyAI upload failed to return URL.")
                                
                                transcript_payload = {"audio_url": upload_url_assembly, "word_details": True} 
                                transcript_post_response = await client_asm.post("https://api.assemblyai.com/v2/transcript", headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}, json=transcript_payload)
                                transcript_post_response.raise_for_status()
                                transcript_id_assembly = transcript_post_response.json().get("id")

                                if not transcript_id_assembly: raise Exception("AssemblyAI failed to return transcript ID.")

                                logger.info(f"[{request_id}] AssemblyAI transcription started (ID: {transcript_id_assembly}). Polling...")
                                transcript_get_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id_assembly}"
                                
                                for _ in range(int(ASSEMBLYAI_API_TIMEOUT / ASSEMBLYAI_POLL_INTERVAL)):
                                    await asyncio.sleep(REPLICATE_POLL_INTERVAL) # Corrected variable
                                    poll_response = await client_asm.get(transcript_get_url, headers={"authorization": ASSEMBLYAI_API_KEY})
                                    poll_response.raise_for_status()
                                    transcript_result = poll_response.json()
                                    status = transcript_result.get("status")
                                    logger.info(f"[{request_id}] AssemblyAI poll status: {status}")
                                    if status == "completed":
                                        generated_files_summary["assembly_ai_raw_transcript"] = transcript_result
                                        srt_content_str = _format_assemblyai_transcript_to_srt(transcript_result) 
                                        logger.info(f"[{request_id}] AssemblyAI transcription completed and SRT formatted.")
                                        break
                                    elif status == "error":
                                        raise Exception(f"AssemblyAI transcription failed: {transcript_result.get('error')}")
                                else:
                                    raise Exception(f"AssemblyAI transcription timed out for ID: {transcript_id_assembly}")
                            except Exception as e_assembly:
                                logger.error(f"[{request_id}] AssemblyAI error: {e_assembly}", exc_info=True)
                                generated_files_summary["errors"].append({"step": current_step, "error": f"AssemblyAI error: {e_assembly}"})
                                srt_content_str = None 
                    else:
                        logger.warning(f"[{request_id}] No valid narration audio found for AssemblyAI transcription.")

                if not srt_content_str and parsed_prompt_data.get("subtitles"):
                    logger.info(f"[{request_id}] Using Claude-generated subtitles.")
                    srt_content_str = await _format_claude_subtitles_to_srt(
                        parsed_prompt_data["subtitles"],
                        generated_files_summary["scenes"], 
                        actual_total_processed_video_duration,
                        request_id
                    )
                    if not srt_content_str:
                        logger.warning(f"[{request_id}] Claude subtitle generation resulted in empty SRT.")
                        generated_files_summary["errors"].append({"step": current_step, "error": "Claude subtitle generation resulted in empty SRT."})

                if not srt_content_str and req.subtitle_script_prompt: # This is from VideoGenerationRequest, not VideoEditRequest's new_script_srt
                    logger.info(f"[{request_id}] Using subtitle script from request parameters.")
                    duration_for_srt = actual_total_processed_video_duration if actual_total_processed_video_duration > 0 else (req.duration or 10.0)
                    srt_content_str = f"1\n00:00:00,000 --> {format_time_srt(duration_for_srt)}\n{req.subtitle_script_prompt}\n"
                    logger.info(f"[{request_id}] Created basic SRT from user script.")

            # Translate subtitles if needed
            if srt_content_str and req.subtitle_target_lang and req.subtitle_target_lang.lower() != (req.subtitle_source_lang or req.narration_lang).lower():
                if DEEPL_API_KEY and hasattr(fastapi_request.app.state, 'deepl_translator') and fastapi_request.app.state.deepl_translator:
                    logger.info(f"[{request_id}] Translating subtitles to {req.subtitle_target_lang}...")
                    try:
                        srt_content_str = await _translate_srt_content(
                            srt_content_str, 
                            req.subtitle_target_lang, 
                            req.subtitle_source_lang or req.narration_lang, 
                            fastapi_request,
                            request_id,
                            generated_files_summary
                        )
                        logger.info(f"[{request_id}] Subtitles translated successfully.")
                    except Exception as e_deepl:
                        logger.error(f"[{request_id}] DeepL translation error: {e_deepl}", exc_info=True)
                        generated_files_summary["errors"].append({"step": "subtitle_translation", "error": str(e_deepl)})
                else:
                    logger.warning(f"[{request_id}] DeepL API key not configured or translator not initialized. Skipping translation.")
                    generated_files_summary["errors"].append({"step": current_step, "error": "DeepL translation skipped due to missing key or client."})
            
            if srt_content_str:
                subtitle_file_path = temp_dir / f"subtitles_{request_id}.srt"
                try:
                    with open(subtitle_file_path, "w", encoding="utf-8") as f:
                        f.write(srt_content_str)
                    generated_files_summary["subtitle_file"] = str(subtitle_file_path.resolve())
                    logger.info(f"[{request_id}] Subtitle file saved: {subtitle_file_path.resolve()}")
                except Exception as e_save_srt:
                    logger.error(f"[{request_id}] Error saving SRT file: {e_save_srt}", exc_info=True)
                    generated_files_summary["errors"].append({"step": current_step, "error": f"Failed to save SRT file: {e_save_srt}"})
            else:
                logger.info(f"[{request_id}] No subtitle content generated or available to process.")
                generated_files_summary["subtitle_file"] = None
        else:
            logger.info(f"[{request_id}] Subtitle generation is disabled by user request.")
            generated_files_summary["subtitle_file"] = None
        save_status(task_id, {"status": "processing", "message": "Subtitle generation step finished.", "current_step": current_step, "details": generated_files_summary})
        
        # 5. BGM Generation
        current_step = "bgm_generation"
        logger.info(f"[{request_id}] Step 5: Generating/Reusing BGM...")
        
        should_regenerate_bgm = not is_editing or \
                                (is_editing and edit_request_params and edit_request_params.edit_target == EditTargetEnum.BGM) or \
                                (req.bgm_enabled and (not generated_files_summary.get("bgm_audio_file") or not Path(generated_files_summary["bgm_audio_file"]).exists()))

        if is_editing and edit_request_params and edit_request_params.edit_target != EditTargetEnum.BGM and \
           original_task_details and original_task_details.get("bgm_audio_file") and Path(original_task_details["bgm_audio_file"]).exists():
            original_bgm_path = Path(original_task_details["bgm_audio_file"])
            new_bgm_path = temp_dir / original_bgm_path.name
            if original_bgm_path != new_bgm_path:
                shutil.copy(original_bgm_path, new_bgm_path)
            generated_files_summary["bgm_audio_file"] = str(new_bgm_path.resolve())
            logger.info(f"[{request_id}] Reusing existing BGM file: {new_bgm_path}")
            should_regenerate_bgm = False
        elif is_editing and edit_request_params and edit_request_params.edit_target == EditTargetEnum.BGM and not req.bgm_enabled:
            logger.info(f"[{request_id}] BGM explicitly disabled by edit request.")
            generated_files_summary["bgm_audio_file"] = None
            should_regenerate_bgm = False

        if req.bgm_enabled and should_regenerate_bgm:
            if STABILITY_API_KEY:
                bgm_prompt_text = req.bgm_prompt # This should be updated by the edit_video endpoint if BGM is the target
                if not bgm_prompt_text: # Fallback if not set by edit_video
                    bgm_prompt_text = parsed_prompt_data.get("bgm", {}).get("description", "calm instrumental background music")
                
                bgm_duration_seconds = int(actual_total_processed_video_duration) if actual_total_processed_video_duration > 0 else (req.duration or 10)
                if bgm_duration_seconds <=0: bgm_duration_seconds = 10 

                logger.info(f"[{request_id}] BGM: Using Stability AI. Prompt: '{bgm_prompt_text}', Duration: {bgm_duration_seconds}s")
                
                data = {
                    "text_prompt": bgm_prompt_text,
                    "duration": str(bgm_duration_seconds),
                    "model": "stable-audio-2.0", 
                    "output_format": "mp3", 
                    "seed": random.randint(0, 4294967295)
                }

                async with httpx.AsyncClient(timeout=STABILITY_API_TIMEOUT) as client:
                    try:
                        response = await client.post(
                            STABILITY_AUDIO_API_URL,
                            headers={
                                "Authorization": f"Bearer {STABILITY_API_KEY}",
                            },
                            data=data 
                        )
                        response.raise_for_status() 
                        
                        content_type = response.headers.get("Content-Type", "audio/mpeg")
                        bgm_file_extension = "mp3"
                        if "wav" in content_type:
                            bgm_file_extension = "wav"
                        
                        bgm_file_path = temp_dir / f"bgm_audio_{request_id}.{bgm_file_extension}"
                        with open(bgm_file_path, "wb") as f:
                            f.write(response.content)
                        generated_files_summary["bgm_audio_file"] = str(bgm_file_path.resolve())
                        logger.info(f"[{request_id}] BGM generated successfully and saved to {bgm_file_path.name}")

                    except httpx.HTTPStatusError as e_stab_audio:
                        error_detail = f"Stability AI BGM generation API error: {e_stab_audio.response.status_code} - {e_stab_audio.response.text}"
                        logger.error(f"[{request_id}] {error_detail}", exc_info=True)
                        generated_files_summary["errors"].append({"step": current_step, "error": error_detail})
                        generated_files_summary["bgm_audio_file"] = None 
                    except Exception as e_stab_audio_generic:
                        error_detail = f"Generic error during Stability AI BGM generation: {e_stab_audio_generic}"
                        logger.error(f"[{request_id}] {error_detail}", exc_info=True)
                        generated_files_summary["errors"].append({"step": current_step, "error": error_detail})
                        generated_files_summary["bgm_audio_file"] = None
            else:
                logger.warning(f"[{request_id}] BGM enabled but STABILITY_API_KEY is not set. Skipping BGM generation.")
                generated_files_summary["errors"].append({"step": current_step, "error": "STABILITY_API_KEY missing for BGM."})
                generated_files_summary["bgm_audio_file"] = None
        elif not req.bgm_enabled:
            logger.info(f"[{request_id}] BGM generation is disabled by user request.")
            generated_files_summary["bgm_audio_file"] = None
        
        save_status(task_id, {"status": "processing", "message": "BGM generation step finished.", "current_step": current_step, "details": generated_files_summary})

        # 6. Video Integration (FFmpeg)
        current_step = "video_integration"
        logger.info(f"[{request_id}] Step 6: Integrating materials with FFmpeg...")
        save_status(task_id, {"status": "processing", "message": "Integrating video components...", "current_step": current_step, "details": generated_files_summary})

        final_video_filename = f"{request_id}_final.{req.output_format}"
        final_output_dir = STATIC_VIDEO_DIR / request_id
        final_output_dir.mkdir(parents=True, exist_ok=True)
        # final_video_output_path_temp = temp_dir / final_video_filename # This was incorrect, should be in final_output_dir for final output
        final_video_output_path = final_output_dir / final_video_filename


        ffmpeg_cmd = [FFMPEG_COMMAND, "-y"]
        
        # Rebuild video_input_options and video_filter_inputs based on generated_files_summary["scenes"]
        # This ensures that reused scene paths are correctly included for FFmpeg
        video_input_options = []
        video_filter_inputs = []
        actual_total_processed_video_duration = 0.0 # Recalculate based on final scene list

        for idx, scene_data in enumerate(generated_files_summary["scenes"]):
            material_path_str = scene_data.get("path")
            if material_path_str and Path(material_path_str).is_file():
                material_path_obj = Path(material_path_str)
                scene_duration = float(scene_data.get("final_duration", default_scene_duration))

                if scene_data.get("material_type") == "image":
                    video_input_options.extend(["-loop", "1", "-r", str(DEFAULT_FPS), "-t", str(scene_duration), "-i", str(material_path_obj.resolve())])
                else: # video file
                    video_input_options.extend(["-i", str(material_path_obj.resolve())])
                
                video_filter_inputs.append(f"[{idx}:v]") # Index based on the order in generated_files_summary["scenes"]
                actual_total_processed_video_duration += scene_duration
            else:
                logger.warning(f"[{request_id}] Scene {idx} material path is missing or invalid for FFmpeg: {material_path_str}. This might cause FFmpeg errors.")
                # Consider adding a black screen placeholder here if a scene is missing
                # For now, we rely on prior steps to ensure valid paths or handle errors.

        ffmpeg_cmd.extend(video_input_options)

        audio_input_options_ffmpeg: List[str] = []
        # audio_filter_inputs_ffmpeg: Dict[str, str] = {} # This was incorrect, ffmpeg input indices are global
        current_ffmpeg_input_idx = len(video_filter_inputs) # Start counting from after video inputs

        valid_narration_paths_ffmpeg = [
            nar_info["path"] 
            for nar_info in generated_files_summary.get("narration_audios", []) 
            if nar_info.get("path") and Path(nar_info["path"]).exists()
        ]

        narration_concat_file_path_ffmpeg_str = None
        narration_stream_ffmpeg_id = None
        if req.narration_enabled and valid_narration_paths_ffmpeg:
            if len(valid_narration_paths_ffmpeg) > 1:
                narration_concat_file_path_ffmpeg = temp_dir / f"narrations_ffmpeg_concat_{request_id}.txt"
                narration_concat_file_path_ffmpeg_str = str(narration_concat_file_path_ffmpeg.resolve())
                with open(narration_concat_file_path_ffmpeg, "w") as f:
                    for path_str in valid_narration_paths_ffmpeg:
                        f.write(f"file '{Path(path_str).resolve().as_posix()}'\n")
                audio_input_options_ffmpeg.extend(["-f", "concat", "-safe", "0", "-i", narration_concat_file_path_ffmpeg_str])
                narration_stream_ffmpeg_id = f"[{current_ffmpeg_input_idx}:a]"
                current_ffmpeg_input_idx += 1
            elif len(valid_narration_paths_ffmpeg) == 1:
                audio_input_options_ffmpeg.extend(["-i", str(Path(valid_narration_paths_ffmpeg[0]).resolve())])
                narration_stream_ffmpeg_id = f"[{current_ffmpeg_input_idx}:a]"
                current_ffmpeg_input_idx += 1
        
        bgm_file_path_str = generated_files_summary.get("bgm_audio_file")
        bgm_stream_ffmpeg_id = None
        if req.bgm_enabled and bgm_file_path_str and Path(bgm_file_path_str).is_file():
            audio_input_options_ffmpeg.extend(["-i", str(Path(bgm_file_path_str).resolve())])
            bgm_stream_ffmpeg_id = f"[{current_ffmpeg_input_index}:a]"
            current_ffmpeg_input_index += 1

        ffmpeg_cmd.extend(audio_input_options_ffmpeg)

        filter_complex_parts = []
        
        if not video_filter_inputs:
            error_message = "No valid video inputs for FFmpeg processing."
            logger.error(f"[{request_id}] {error_message}")
            generated_files_summary["errors"].append({"step": current_step, "error": error_message})
            raise ValueError(error_message)

        video_processing_chain = ""
        if len(video_filter_inputs) > 1:
            concat_inputs_str = "".join(video_filter_inputs)
            video_processing_chain = f"{concat_inputs_str}concat=n={len(video_filter_inputs)}:v=1:a=0[vconcat_out];"
            current_video_stream_label = "[vconcat_out]"
        else: 
            # For a single video input, no concat, but still need to pass it through for scaling/padding
            video_processing_chain = f"{video_filter_inputs[0]}null[vconcat_out];" # Use null filter to just label it
            current_video_stream_label = "[vconcat_out]"


        target_w, target_h = parse_resolution(req.resolution)
        scale_pad_filter = f"{current_video_stream_label}fps={DEFAULT_FPS},format=yuv420p,scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:eval=frame,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black[vscaled_out]"
        
        video_processing_chain += scale_pad_filter 
        filter_complex_parts.append(video_processing_chain)
        
        final_video_stream_for_map = "[vscaled_out]"

        subtitle_file_path = generated_files_summary.get("subtitle_file")
        if req.subtitles_enabled and subtitle_file_path and Path(subtitle_file_path).exists():
            logger.info(f"[{request_id}] Adding subtitles from: {subtitle_file_path} to FFmpeg command.")
            escaped_subtitle_path = str(Path(subtitle_file_path).resolve()).replace('\\', '/').replace(':', r'\\:')
            
            force_style_parts = []
            if req.subtitle_font_name: force_style_parts.append(f"FontName='{req.subtitle_font_name}'")
            if req.subtitle_font_size: force_style_parts.append(f"FontSize={req.subtitle_font_size}")
            if req.subtitle_primary_color: force_style_parts.append(f"PrimaryColour={req.subtitle_primary_color}")
            if req.subtitle_outline_color: force_style_parts.append(f"OutlineColour={req.subtitle_outline_color}")
            if req.subtitle_background_color:
                force_style_parts.append(f"BackColour={req.subtitle_background_color}")
                force_style_parts.append("BorderStyle=3") 
            if req.subtitle_alignment is not None: force_style_parts.append(f"Alignment={req.subtitle_alignment}")
            if req.subtitle_margin_v is not None: force_style_parts.append(f"MarginV={req.subtitle_margin_v}")
            
            style_string = ",".join(force_style_parts)
            logger.info(f"[{request_id}] Subtitle force_style string: '{style_string}'")

            subtitle_filter_string = f"subtitles=filename='{escaped_subtitle_path}'"
            if style_string:
                subtitle_filter_string += f":force_style='{style_string}'"
            
            filter_complex_parts.append(f"{final_video_stream_for_map}{subtitle_filter_string}[vout]")
            final_video_map_label = "[vout]"
        else:
            if filter_complex_parts and final_video_stream_for_map == "[vscaled_out]" and not filter_complex_parts[-1].endswith("[vout]"):
                 filter_complex_parts[-1] = filter_complex_parts[-1].replace("[vscaled_out]", "[vout]")
            elif not filter_complex_parts and video_filter_inputs: 
                 filter_complex_parts.append(f"{video_filter_inputs[0]}copy[vout]") 
            final_video_map_label = "[vout]"


        final_audio_map_label = None
        audio_processing_filters = []
        fade_duration_seconds = 3.0
        video_duration_for_fade = actual_total_processed_video_duration if actual_total_processed_video_duration > 0 else (req.duration or 10.0)
        fade_start_time = max(0, video_duration_for_fade - fade_duration_seconds)
        logger.info(f"[{request_id}] Audio fade out: start_time={fade_start_time}, duration={fade_duration_seconds}, video_duration_for_fade={video_duration_for_fade}")


        if narration_stream_ffmpeg_id and bgm_stream_ffmpeg_id:
            audio_processing_filters.append(f"{bgm_stream_ffmpeg_id}volume=0.3,afade=t=out:st={fade_start_time}:d={fade_duration_seconds}[bgm_faded_out];"
                                            f"{narration_stream_ffmpeg_id}afade=t=out:st={fade_start_time}:d={fade_duration_seconds}[nar_faded_out];"
                                            f"[nar_faded_out][bgm_faded_out]amix=inputs=2:duration=longest[aout]")
            final_audio_map_label = "[aout]"
        elif narration_stream_ffmpeg_id:
            audio_processing_filters.append(f"{narration_stream_ffmpeg_id}afade=t=out:st={fade_start_time}:d={fade_duration_seconds}[aout]")
            final_audio_map_label = "[aout]"
        elif bgm_stream_ffmpeg_id:
            audio_processing_filters.append(f"{bgm_stream_ffmpeg_id}volume=0.3,afade=t=out:st={fade_start_time}:d={fade_duration_seconds}[aout]")
            final_audio_map_label = "[aout]"

        if audio_processing_filters:
            filter_complex_parts.extend(audio_processing_filters)
            logger.info(f"[{request_id}] Audio filter parts: {audio_processing_filters}")


        if filter_complex_parts:
            ffmpeg_cmd.extend(["-filter_complex", ";".join(processed_filter_parts for processed_filter_parts in filter_complex_parts if processed_filter_parts)])
        
        ffmpeg_cmd.extend(["-map", final_video_map_label])
        if final_audio_map_label:
            ffmpeg_cmd.extend(["-map", final_audio_map_label])
        else:
            ffmpeg_cmd.append("-an")

        ffmpeg_cmd.extend([
            "-c:v", "libx264", 
            "-preset", "medium", 
            "-crf", "23", 
            "-c:a", "aac", 
            "-b:a", "192k",
            "-movflags", "+faststart",
            # "-shortest", # Replaced by -t below for more precise control
        ])
        if actual_total_processed_video_duration > 0:
             ffmpeg_cmd.extend(["-t", str(actual_total_processed_video_duration)])
        
        ffmpeg_cmd.append(str(final_video_output_path_temp)) # Output to temp first

        logger.info(f"[{request_id}] Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        success, ffmpeg_stdout, ffmpeg_stderr = await run_ffmpeg_command_async(ffmpeg_cmd, request_id)
        generated_files_summary["ffmpeg_final_video_log"] = {
            "stdout": ffmpeg_stdout,
            "stderr": ffmpeg_stderr,
            "command": " ".join(ffmpeg_cmd)
        }

        if success:
            # Move to static serving directory
            permanent_video_path = final_output_dir / final_video_filename
            shutil.move(str(final_video_output_path_temp), str(permanent_video_path))
            final_video_successfully_moved = True
            
            generated_files_summary["final_video_path"] = str(permanent_video_path.resolve())
            generated_files_summary["final_video_url"] = f"/static/{STATIC_VIDEO_DIR_BASE.name}/{request_id}/{final_video_filename}"
            logger.info(f"[{request_id}] Final video moved to {permanent_video_path} and URL {generated_files_summary['final_video_url']} prepared.")
            save_status(task_id, {"status": "completed", "message": "Video generation complete.", "result_url": generated_files_summary["final_video_url"], "current_step": current_step, "details": generated_files_summary})

        else:
            err_msg = f"FFmpeg final video processing failed. See logs for task_id {request_id}."
            logger.error(f"[{request_id}] {err_msg} STDERR: {ffmpeg_stderr}")
            generated_files_summary["errors"].append({"step": current_step, "error": err_msg, "ffmpeg_stderr": ffmpeg_stderr})
            save_status(task_id, {"status": "error", "message": err_msg, "current_step": current_step, "details": generated_files_summary})
            raise ValueError(err_msg)

    except ValueError as ve: 
        error_message = f"Error in step '{current_step}': {str(ve)}"
        logger.error(f"[{request_id}] {error_message}", exc_info=True)
        generated_files_summary["errors"].append({"step": current_step, "error": str(ve)})
        save_status(task_id, {"status": "error", "message": error_message, "current_step": current_step, "details": generated_files_summary})
    except httpx.HTTPStatusError as hse:
        error_message = f"HTTP error during {current_step}: {hse.response.status_code} - {hse.response.text}"
        logger.error(f"[{request_id}] {error_message}", exc_info=True)
        generated_files_summary["errors"].append({"step": current_step, "error": error_message})
        save_status(task_id, {"status": "error", "message": error_message, "current_step": current_step, "details": generated_files_summary})
    except Exception as e:
        error_message = f"An unexpected error occurred during {current_step}: {str(e)}"
        logger.error(f"[{request_id}] {error_message}", exc_info=True)
        generated_files_summary["errors"].append({"step": current_step, "error": error_message})
        save_status(task_id, {"status": "error", "message": error_message, "current_step": current_step, "details": generated_files_summary})
    finally:
        final_status_data = get_status_from_file(task_id) or {} 
        final_status_data.update({
            "status": final_status_data.get("status", "error" if generated_files_summary["errors"] else "completed"), 
            "message": final_status_data.get("message", f"Processing finished. Errors: {len(generated_files_summary['errors'])}" if generated_files_summary["errors"] else "Video generation completed successfully."),
            "result_url": generated_files_summary.get("final_video_url"), 
            "details": generated_files_summary,
            "error_details": json.dumps(generated_files_summary["errors"]) if generated_files_summary["errors"] else None
        })
        save_status(task_id, final_status_data)
        
        if CLEANUP_TEMP_FILES:
            final_video_path_obj = Path(generated_files_summary.get("final_video_path", ""))
            if final_video_successfully_moved or (generated_files_summary["errors"] and not (final_video_path_obj.exists() and temp_dir in final_video_path_obj.parents)):
                 logger.info(f"[{request_id}] Cleaning up temporary directory: {temp_dir.resolve()}")
                 shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                 logger.info(f"[{request_id}] Temporary files for video generation kept at {temp_dir.resolve()}.")
            
    return generated_files_summary


@router.post("/generate_from_text/", response_model=VideoGenerationResponse)
async def generate_video_from_text_endpoint(
    req: VideoGenerationRequest, 
    fastapi_request: Request, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user) 
):
    logger.info(f"User {current_user.email} (ID: {current_user.id}) initiated video generation with prompt: '{req.prompt[:50]}...'")
    task_id = str(uuid.uuid4())
    
    task_temp_dir = TEMP_DIR / task_id 
    task_temp_dir.mkdir(parents=True, exist_ok=True) 
    
    initial_status = {
        "task_id": task_id,
        "status": "queued", 
        "message": "Video generation task received and queued.",
        "details": { 
            "request_id": task_id,
            "params": req.model_dump(),
            "claude_analysis": None,
            "scenes": [], 
            "narration_audios": [], 
            "subtitle_file": None, 
            "bgm_audio_file": None, 
            "final_video_path": None,
            "final_video_url": None,
            "errors": [],
            "summary_file_path": str(task_temp_dir / f"generation_summary_{task_id}.json")
        }
    }
    save_status(task_id, initial_status)

    background_tasks.add_task(create_video_from_text_pipeline, req, fastapi_request, task_id, None, None)
    
    return VideoGenerationResponse(
        message="Video generation process started in the background.",
        task_id=task_id,
        status_url=str(fastapi_request.url_for('get_task_status_endpoint', task_id=task_id)),
        debug_data_url=str(fastapi_request.url_for('serve_generated_video_debug_file', task_id=task_id, filename=f"generation_summary_{task_id}.json"))
    )

@router.post("/edit/{original_task_id}", response_model=VideoGenerationResponse, summary="Edit a previously generated video")
async def edit_video(
    original_task_id: str,
    edit_request: VideoEditRequest, # This is the Pydantic model defined earlier
    background_tasks: BackgroundTasks,
    fastapi_request: Request, 
    current_user: User = Depends(get_current_active_user)
):
    logger.info(f"User {current_user.email} (ID: {current_user.id}) initiated video edit for original task ID: {original_task_id}")
    logger.info(f"Edit request details: {edit_request.model_dump_json(indent=2, exclude_none=True)}")

    original_status_file = TEMP_DIR / original_task_id / f"{original_task_id}_status.json"
    if not original_status_file.exists():
        logger.error(f"Original task status file not found: {original_status_file}")
        raise HTTPException(status_code=404, detail=f"Original task {original_task_id} not found or its data is missing.")

    try:
        with open(original_status_file, "r", encoding="utf-8") as f:
            original_task_status = json.load(f)
        original_task_details = original_task_status.get("details")
        if not original_task_details:
            logger.error(f"Original task details not found in status file for {original_task_id}")
            raise HTTPException(status_code=500, detail="Original task details not found.")
        original_params_dict = original_task_details.get("params")
        if not original_params_dict:
            logger.error(f"Original task parameters not found in status file for {original_task_id}")
            raise HTTPException(status_code=500, detail="Original task parameters not found.")

    except json.JSONDecodeError:
        logger.error(f"Error decoding original task status file for {original_task_id}")
        raise HTTPException(status_code=500, detail="Could not load original task data due to JSON error.")
    except Exception as e:
        logger.error(f"Error reading original task data for {original_task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load original task data: {e}")

    # Create a new VideoGenerationRequest by updating the original params with edit_request fields
    updated_params_dict = original_params_dict.copy()

    # Update fields from VideoGenerationRequest based on edit_request
    for field_name, field_value in edit_request.model_dump(exclude_unset=True).items():
        if field_name not in ["edit_target", "target_indices", "new_prompt", "new_value_numeric", "new_value_string", "new_script_srt"]:
            if field_value is not None: # Only update if a new value is provided
                updated_params_dict[field_name] = field_value
    
    # Handle specific edit_target logic that might override general params
    if edit_request.edit_target == EditTargetEnum.NARRATION_FULL_SCRIPT and edit_request.new_prompt:
        updated_params_dict["narration_script_prompt"] = edit_request.new_prompt
    elif edit_request.edit_target == EditTargetEnum.SUBTITLE_FULL_SCRIPT and edit_request.new_script_srt:
        updated_params_dict["subtitle_script_prompt"] = edit_request.new_script_srt # This will be used to create the SRT
    elif edit_request.edit_target == EditTargetEnum.BGM and edit_request.new_prompt:
        updated_params_dict["bgm_prompt"] = edit_request.new_prompt

    # Ensure subtitles are enabled if style is being edited
    if edit_request.edit_target == EditTargetEnum.SUBTITLE_STYLE:
        updated_params_dict["subtitles_enabled"] = True


    try:
        edited_video_req_model = VideoGenerationRequest(**updated_params_dict)
    except Exception as e:
        logger.error(f"Error creating VideoGenerationRequest from merged data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters for video generation after edit: {e}")

    new_task_id = str(uuid.uuid4())
    new_task_temp_dir = TEMP_DIR / new_task_id
    new_task_temp_dir.mkdir(parents=True, exist_ok=True)
    
    initial_status_for_edit = {
        "task_id": new_task_id,
        "original_task_id_edited": original_task_id,
        "status": "queued",
        "message": "Video edit task received and queued for regeneration.",
        "edit_request_details": edit_request.model_dump(exclude_none=True),
        "details": { 
            "request_id": new_task_id,
            "params": edited_video_req_model.model_dump(), 
            "claude_analysis": None, 
            "scenes": [], 
            "narration_audios": [], 
            "subtitle_file": None, 
            "bgm_audio_file": None, 
            "final_video_path": None,
            "final_video_url": None,
            "errors": [],
            "summary_file_path": str(new_task_temp_dir / f"generation_summary_{new_task_id}.json")
        }
    }
    save_status(new_task_id, initial_status_for_edit)
    
    background_tasks.add_task(create_video_from_text_pipeline, edited_video_req_model, fastapi_request, new_task_id, original_task_details, edit_request)

    return VideoGenerationResponse(
        message=f"Video edit task initiated. Original task ID: {original_task_id}. New task ID for edited video: {new_task_id}.",
        task_id=new_task_id,
        status_url=str(fastapi_request.url_for('get_task_status_endpoint', task_id=new_task_id)),
        debug_data_url=str(fastapi_request.url_for('serve_generated_video_debug_file', task_id=new_task_id, filename=f"generation_summary_{new_task_id}.json"))
    )


@router.get("/status/{task_id}", response_model=TaskStatus) 
async def get_task_status_endpoint(task_id: str, request: Request):
    status_data = get_status_from_file(task_id)
    if not status_data:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found or task not yet started.")
    
    data_to_return = {
        "task_id": status_data.get("task_id", task_id),
        "status": status_data.get("status", "unknown"),
        "message": status_data.get("message"),
        "progress": status_data.get("progress"),
        "result_url": status_data.get("result_url"), 
        "details": status_data.get("details"),
        "error_details": status_data.get("error_details") 
    }
    if status_data.get("details") and status_data["details"].get("summary_file_path"):
        summary_file_name = Path(status_data["details"]["summary_file_path"]).name
        data_to_return["debug_data_url"] = str(request.url_for('serve_generated_video_debug_file', task_id=task_id, filename=summary_file_name))

    return TaskStatus(**data_to_return)


@router.get(f"/{STATIC_VIDEO_DIR_BASE.name}/{{task_id}}/{{filename:path}}")
async def serve_generated_video_debug_file(task_id: str, filename: str):
    file_path = STATIC_VIDEO_DIR / task_id / filename
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    if filename.endswith(".json"):
        return FileResponse(path=str(file_path), media_type="application/json")
    elif filename.endswith((".mp4", ".mov", ".webm", ".avi")): 
        return FileResponse(path=str(file_path), media_type=f"video/{Path(filename).suffix[1:]}")
    else:
        return FileResponse(path=str(file_path))

if __name__ != "__main__": 
    try:
        from main import get_claude_response, IndividualAIResponse, app as main_app
        # from models import User 
        # from dependencies import get_current_active_user 
    except ImportError as e:
        logger.warning(f"Could not import from main/models/dependencies: {e}. Using placeholders.")
        
        class IndividualAIResponse:
            def __init__(self, response: Optional[str], error: Optional[str] = None):
                self.response = response
                self.error = error

        async def get_claude_response(request: Request, prompt_text: str, system_instruction: str, model: str) -> IndividualAIResponse:
            logger.error("Mocked get_claude_response called.")
            mock_claude_output = {
                "scenes": [{"scene_description": "A mock beautiful sunset.", "duration_seconds": 5, "narration_segment_indices": [0], "subtitle_indices": [0]},
                           {"scene_description": "Mock city traffic.", "duration_seconds": 5, "narration_segment_indices": [1], "subtitle_indices": [1]}],
                "narration": {"full_script": "Mock narration.", "segments": [{"text": "Mock segment 1."}, {"text": "Mock segment 2."}]},
                "subtitles": [{"text": "Mock sub 1.", "timing_instructions": "scene 1"}, {"text": "Mock sub 2.", "timing_instructions": "scene 2"}],
                "bgm": {"description": "Mock upbeat electronic music."}
            }
            return IndividualAIResponse(response=json.dumps(mock_claude_output))

        class User: 
            id: int = 1
            email: str = "test@example.com"
        
        async def get_current_active_user() -> User: 
            return User()

        class MockAppState:
            def __init__(self):
                self.deepl_translator = None 
        
        class MockApp:
            def __init__(self):
                self.state = MockAppState()

        main_app = MockApp()

    REPLICATE_MAX_POLL_ATTEMPTS = REPLICATE_MAX_POLL_ATTEMPTS if 'REPLICATE_MAX_POLL_ATTEMPTS' in globals() else 30
    FFMPEG_COMMAND = FFMPEG_COMMAND if 'FFMPEG_COMMAND' in globals() else "ffmpeg"

else: 
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    
    STATIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path("static").exists():
        Path("static").mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static_files_root")


    class MockAppState:
        def __init__(self):
            self.deepl_translator = None 
    app.state = MockAppState()
    
    logger.info(f"Serving static files from: {STATIC_VIDEO_DIR_BASE}")
    logger.info(f"Static files URL prefix: /static/{STATIC_VIDEO_DIR_BASE.name}")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

```
```python
# Configuration settings for the FastAPI application

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_AI_KEY") # Corrected from STEDDY_API_KEY
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") 
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") 

# API Endpoints and Model Identifiers
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_ZEROSCOPE_MODEL_XL = "anotherjesse/zeroscope-v2-xl:9f7476737190e1a712580adfd5446408f14b2de0e6e8e168d68f2029fc221216"
REPLICATE_ZEROSCOPE_MODEL_576W = "anotherjesse/zeroscope-v2-576w:1c8f6c34d800a8054187871f754559085323598320e960e699500244a8386153" # Example, confirm if this is the one you want as default or an option
STABILITY_TEXT_TO_IMAGE_API_URL_BASE = "https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
STABILITY_DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0" # Example, use the latest or preferred engine
STABILITY_AUDIO_API_URL = "https://api.stability.ai/v1/generation/stable-audio-generate-v1"

# Polling and Timeout settings for API calls
REPLICATE_POLL_INTERVAL = 10  # seconds
REPLICATE_API_TIMEOUT = 300.0  # seconds
STABILITY_API_TIMEOUT = 180.0  # seconds
ELEVENLABS_API_TIMEOUT = 60.0  # seconds
ASSEMBLYAI_POLL_INTERVAL = 5   # seconds
ASSEMBLYAI_API_TIMEOUT = 300.0 # seconds for transcription

# Video Processing Defaults
DEFAULT_FPS = 24 # Changed from 25 to align with common video frame rates

# Directory Settings
TEMP_DIR_BASE = "temp_files"  # Base directory for temporary files during processing
STATIC_DIR = "static" # General static directory
STATIC_VIDEO_DIR_NAME = "generated_videos" # Subdirectory for generated videos
STATIC_VIDEO_DIR = Path(STATIC_DIR) / STATIC_VIDEO_DIR_NAME # Full path to static videos directory

# File Management
CLEANUP_TEMP_FILES = True  # Set to False for debugging to keep intermediate files

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
}

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

# Other constants
REPLICATE_MAX_POLL_ATTEMPTS = int(REPLICATE_API_TIMEOUT / REPLICATE_POLL_INTERVAL) if REPLICATE_POLL_INTERVAL > 0 else 30
FFMPEG_COMMAND = "ffmpeg" # Ensure ffmpeg is in PATH or provide full path

# Ensure static directories exist
Path(TEMP_DIR_BASE).mkdir(parents=True, exist_ok=True)
STATIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# API Key checks (optional, for developer awareness during startup)
# if not STABILITY_API_KEY:
#     print("Warning: STABILITY_API_KEY is not set in the environment variables.")
# if not REPLICATE_API_TOKEN:
#     print("Warning: REPLICATE_API_TOKEN is not set in the environment variables.")
# if not ELEVENLABS_API_KEY:
#     print("Warning: ELEVENLABS_API_KEY is not set in the environment variables.")
# if not ASSEMBLYAI_API_KEY:
#     print("Warning: ASSEMBLYAI_API_KEY is not set in the environment variables.")
# if not DEEPL_API_KEY:
#     print("Warning: DEEPL_API_KEY is not set in the environment variables.")

```
