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
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip, ImageClip, 
    concatenate_videoclips, TextClip, CompositeVideoClip
)
from moviepy.video.fx.all import resize as moviepy_resize # Renamed to avoid conflict
from moviepy.config import change_settings
from PIL import Image # Import Pillow

# Import config variables from config.py
try:
    from config import (
        REPLICATE_API_TOKEN, STABILITY_API_KEY, ELEVENLABS_API_KEY, ASSEMBLYAI_API_KEY, DEEPL_API_KEY,
        REPLICATE_API_URL, REPLICATE_ZEROSCOPE_MODEL, STABILITY_TEXT_TO_IMAGE_API_URL_BASE,
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
    REPLICATE_ZEROSCOPE_MODEL = "anotherjesse/zeroscope-v2-xl:9f7476737190e1a712580adfd5446408f14b2de0e6e8e168d68f2029fc221216"
    STABILITY_TEXT_TO_IMAGE_API_URL_BASE = "https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
    STABILITY_DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0"
    STABILITY_AUDIO_API_URL = "https://api.stability.ai/v1/generation/stable-audio-generate-v1" 
    REPLICATE_POLL_INTERVAL = 10
    REPLICATE_API_TIMEOUT = 300.0
    STABILITY_API_TIMEOUT = 120.0
    ELEVENLABS_API_TIMEOUT = 60.0
    ASSEMBLYAI_POLL_INTERVAL = 5
    ASSEMBLYAI_API_TIMEOUT = 300.0
    DEFAULT_FPS = 25
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
    subtitle_script_prompt: Optional[str] = Field(None, description="Prompt for Claude to generate subtitle texts.") # Added for user-provided subtitle script
    subtitle_source_lang: Optional[str] = Field(None, description="Source language of subtitles if translation is needed, defaults to narration_lang.")
    subtitle_target_lang: Optional[str] = Field(None, description="Target language for subtitle translation (e.g., EN, ES).")
    bgm_enabled: bool = Field(True, description="Enable or disable background music.")
    bgm_prompt: Optional[str] = Field(None, description="Prompt describing the desired BGM.")
    output_format: str = Field("mp4", description="Output video format (e.g., mp4, mov, webm).")
    video_quality: str = Field("1080p", description="Desired video quality (e.g., 720p, 1080p, 4k).")


class VideoGenerationResponse(BaseModel):
    message: str
    task_id: str
    status_url: str
    video_url: Optional[str] = None
    debug_data_url: Optional[str] = None
    error: Optional[str] = None


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

async def _format_assemblyai_transcript_to_srt(transcript_data: Dict, max_line_len: int = 38, max_duration_s: float = 5.0) -> str:
    srt_entries = []
    if not transcript_data or "words" not in transcript_data:
        return ""

    words = transcript_data["words"]
    if not words:
        return ""

    entry_counter = 1
    current_line = ""
    line_start_time_ms = words[0]["start"]

    for i, word_info in enumerate(words):
        word_text = word_info["text"]
        word_end_time_ms = word_info["end"]

        if not current_line: # First word of a new line
            current_line = word_text
            line_start_time_ms = word_info["start"]
        else:
            # Check if adding the current word exceeds max_line_len or max_duration_s
            if len(current_line + " " + word_text) > max_line_len or \
               (word_end_time_ms - line_start_time_ms) / 1000.0 > max_duration_s:
                # Finalize current line
                srt_entries.append(f"{entry_counter}\n{format_time_srt(line_start_time_ms / 1000.0)} --> {format_time_srt(words[i-1]['end'] / 1000.0)}\n{current_line}")
                entry_counter += 1
                # Start a new line with the current word
                current_line = word_text
                line_start_time_ms = word_info["start"]
            else:
                current_line += " " + word_text
        
        # If it's the last word, add the current entry
        if i == len(words) - 1:
            srt_entries.append(f"{entry_counter}\n{format_time_srt(line_start_time_ms / 1000.0)} --> {format_time_srt(word_end_time_ms / 1000.0)}\n{current_line}")
            
    return "\n\n".join(srt_entries)


async def _format_claude_subtitles_to_srt(claude_subtitles_data: List[Dict], scenes_data: List[Dict], total_video_duration: float, request_id: str) -> str:
    srt_entries = []
    entry_counter = 1
    current_timeline_pos = 0.0 

    if not claude_subtitles_data: return ""

    scene_start_times = {}
    processed_scene_durations_for_subtitles = []
    cumulative_time = 0.0
    for i, scene in enumerate(scenes_data):
        scene_start_times[i] = cumulative_time
        # Use 'final_duration' if available from video processing, otherwise 'duration_seconds' from Claude
        duration = scene.get('final_duration') 
        if duration is None: # Fallback if final_duration isn't set yet (should be by this point)
            duration = scene.get("duration_seconds", 5.0) 
        try: 
            duration = float(duration)
        except (ValueError, TypeError): 
            logger.warning(f"[{request_id}] Invalid duration for scene {i} ('{scene.get('duration_seconds')}') for subtitles. Using default 5s.")
            duration = 5.0
        processed_scene_durations_for_subtitles.append(duration)
        cumulative_time += duration
    
    effective_total_duration = total_video_duration or cumulative_time or (len(claude_subtitles_data) * 3.0)

    for i, sub_entry in enumerate(claude_subtitles_data):
        text = sub_entry.get("text")
        if not text: continue

        start_time_sec = current_timeline_pos 
        estimated_duration = max(2.0, min(len(text.split()) * 0.4, 7.0)) # Default duration logic

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
    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3 and "-->" in lines[1]:
            index = lines[0]
            timestamp = lines[1]
            text_to_translate = "\n".join(lines[2:])
            
            dl_target_lang = DEEPL_LANG_MAP.get(target_lang.lower())
            dl_source_lang = DEEPL_LANG_MAP.get(source_lang.lower()) if source_lang else None
            if not dl_target_lang:
                logger.error(f"[{request_id}] DeepL target language code for '{target_lang}' not found.")
                generated_files_summary_ref["errors"].append({"step": "subtitle_translation", "error": f"DeepL unsupported target language: {target_lang}"})
                translated_lines.append(block) # Append original block if translation fails
                continue

            try:
                # DeepL API might have issues with batch translation of SRT lines directly,
                # so translating line by line within a block if necessary, or as a whole text.
                # For simplicity, we'll translate the whole text content of the block.
                translated_text = deepl_translator.translate_text(
                    text_to_translate, 
                    target_lang=dl_target_lang,
                    source_lang=dl_source_lang
                ).text
                translated_lines.append(f"{index}\n{timestamp}\n{translated_text}")
            except Exception as e:
                logger.error(f"[{request_id}] DeepL translation error for text '{text_to_translate}': {e}")
                generated_files_summary_ref["errors"].append({"step": "subtitle_translation", "error": f"DeepL translation failed for a segment: {str(e)}"})
                translated_lines.append(block) # Append original block on error
        else:
            translated_lines.append(block) # Keep non-subtitle blocks (like empty lines between entries)
            
    logger.info(f"[{request_id}] Subtitle translation to {target_lang} processed.")
    return "\n\n".join(translated_lines)

async def create_video_from_text_pipeline(req: VideoGenerationRequest, fastapi_request: Request, task_id: str):
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
    current_step = "initializing"
    parsed_prompt_data = {} 
    actual_total_scene_duration = 0.0
    final_video_successfully_moved = False

    try:
        save_status(task_id, {"status": "processing", "message": "Video generation process initiated.", "current_step": current_step, "details": generated_files_summary})

        # 1. Claude Prompt Analysis
        current_step = "claude_analysis"
        logger.info(f"[{request_id}] Step 1: Analyzing prompts with Claude...")
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
        if req.subtitle_script_prompt: claude_user_prompt_parts.append(f"Subtitle instructions: {req.subtitle_script_prompt}")
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
            json_str = json_match.group(1) or json_match.group(2)
            parsed_prompt_data = json.loads(json_str)
            generated_files_summary["claude_analysis"] = parsed_prompt_data
            logger.info(f"[{request_id}] Claude analysis successful.")
        except (json.JSONDecodeError, ValueError) as e:
            err_msg = f"Failed to parse JSON from Claude: {e}. Response: {claude_response_obj.response[:500]}"
            generated_files_summary["errors"].append({"step": current_step, "error": err_msg})
            raise ValueError(err_msg)
        save_status(task_id, {"status": "processing", "message": "Claude analysis complete.", "current_step": current_step, "details": generated_files_summary})

        # 2. Video Material Generation
        current_step = "video_material_generation"
        logger.info(f"[{request_id}] Step 2: Generating video/image materials...")
        scenes_plan = parsed_prompt_data.get("scenes", [])
        if not scenes_plan:
            scenes_plan = [{"scene_description": req.prompt, "duration_seconds": req.duration or 10}]
            logger.warning(f"[{request_id}] No scenes from Claude. Using main prompt for one scene.")
            if generated_files_summary["claude_analysis"] is None: generated_files_summary["claude_analysis"] = {}
            if "scenes" not in generated_files_summary["claude_analysis"]: generated_files_summary["claude_analysis"]["scenes"] = []
            generated_files_summary["claude_analysis"]["scenes"] = scenes_plan
        
        target_w_req, target_h_req = parse_resolution(req.resolution)
        
        video_input_options: List[str] = [] 
        video_filter_inputs: List[str] = []
        actual_total_processed_video_duration = 0.0
        
        async with httpx.AsyncClient() as client:
            for i, scene_info_claude in enumerate(scenes_plan):
                scene_summary_entry = {
                    "scene_index": i, 
                    "description_from_claude": scene_info_claude.get("scene_description"),
                    "duration_seconds_claude": scene_info_claude.get("duration_seconds"),
                    "path": None, 
                    "source_api": None, 
                    "error": None,
                    "final_duration": None # Will store the actual duration used in ffmpeg
                }
                scene_desc = scene_info_claude.get("scene_description", f"Default scene prompt for scene {i+1}")
                
                try:
                    scene_duration_seconds = float(scene_info_claude.get("duration_seconds", 5.0))
                    if scene_duration_seconds <= 0: scene_duration_seconds = 1.0 # Minimum duration
                except (ValueError, TypeError):
                    scene_duration_seconds = req.duration / len(scenes_plan) if scenes_plan else 5.0
                    scene_duration_seconds = max(1.0, scene_duration_seconds) # Ensure minimum duration
                    logger.warning(f"[{request_id}] Invalid or missing duration for scene {i}, using calculated/default: {scene_duration_seconds}s.")
                scene_summary_entry["final_duration"] = scene_duration_seconds


                if REPLICATE_API_TOKEN:
                    logger.info(f"[{request_id}] Scene {i+1}: Attempting Replicate Zeroscope for '{scene_desc[:30]}...'")
                    try:
                        num_frames = max(16, min(int(scene_duration_seconds * DEFAULT_FPS), 60)) 
                        rep_payload = {"version": REPLICATE_ZEROSCOPE_MODEL, "input": {"prompt": f"{scene_desc}, cinematic, high quality, {req.video_quality}", "num_frames": num_frames, "width": 1024, "height": 576, "fps": DEFAULT_FPS}}
                        
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
                                        scene_summary_entry["path"] = str(scene_file_path.resolve()); scene_summary_entry["source_api"] = "replicate_zeroscope_v2_xl"
                                        logger.info(f"[{request_id}] Scene {i+1} Replicate success: {scene_file_path.name}")
                                    else: scene_summary_entry["error"] = "Replicate succeeded but no output URL."
                                    break 
                                elif result_data.get("status") == "failed": scene_summary_entry["error"] = f"Replicate prediction failed: {result_data.get('error', 'Unknown error')}"; break
                            else: scene_summary_entry["error"] = "Replicate prediction polling timed out."
                        else: scene_summary_entry["error"] = "Failed to start Replicate prediction."
                    except Exception as e_replicate:
                        logger.warning(f"[{request_id}] Scene {i+1} Replicate API error: {e_replicate}", exc_info=True)
                        scene_summary_entry["error"] = (scene_summary_entry.get("error","") + f"; Replicate API error: {str(e_replicate)[:100]}").strip("; ")
                elif not REPLICATE_API_TOKEN:
                    logger.info(f"[{request_id}] Scene {i+1}: REPLICATE_API_TOKEN not set. Skipping Replicate.")
                    scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Replicate token missing").strip("; ") if scene_summary_entry.get("error") else "Replicate token missing"


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
                                logger.info(f"[{request_id}] Scene {i+1} Stability image success: {scene_file_path_stab.name}")
                            else: scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability: No image data.").strip("; ")
                        else: scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability: No artifacts.").strip("; ")
                    except Exception as e_stability:
                        logger.warning(f"[{request_id}] Scene {i+1} Stability API error: {e_stability}", exc_info=True)
                        scene_summary_entry["error"] = (scene_summary_entry.get("error","") + f"; Stability API error: {str(e_stability)[:100]}").strip("; ")
                elif not STABILITY_API_KEY and not scene_summary_entry["path"]:
                     logger.info(f"[{request_id}] Scene {i+1}: STABILITY_API_KEY not set. Skipping Stability AI.")
                     scene_summary_entry["error"] = (scene_summary_entry.get("error","") + "; Stability API key not configured.").strip("; ") if scene_summary_entry.get("error") else "Stability API key not configured."

                material_path = scene_summary_entry.get("path")
                if material_path and Path(material_path).is_file():
                    material_path_obj = Path(material_path)
                    scene_summary_entry["material_type"] = "video" if material_path_obj.suffix.lower() in ['.mp4', '.webm', '.mov', '.avi'] else "image"
                    
                    if scene_summary_entry["material_type"] == "image":
                        video_input_options.extend(["-loop", "1", "-r", str(DEFAULT_FPS), "-t", str(scene_duration_seconds), "-i", str(material_path_obj.resolve())])
                    else: # video file
                        video_input_options.extend(["-i", str(material_path_obj.resolve())])
                    
                    video_filter_inputs.append(f"[{len(video_input_options) // 2 -1}:v]") # Corrected indexing for inputs
                    actual_total_processed_video_duration += scene_duration_seconds
                elif not scene_summary_entry["error"]:
                     scene_summary_entry["error"] = "All material generation attempts failed or were skipped for this scene."
                
                generated_files_summary["scenes"].append(scene_summary_entry)
                if scene_summary_entry.get("error"): 
                    logger.error(f"[{request_id}] Scene {i+1} final error: {scene_summary_entry['error']}")
                    generated_files_summary["errors"].append({"step": current_step, "scene_index": i, "error": scene_summary_entry['error']})
                save_status(task_id, {"status": "processing", "message": f"Generated material for scene {i+1}/{len(scenes_plan)}", "current_step": current_step, "details": generated_files_summary})
                await asyncio.sleep(0.1) # Small delay
        logger.info(f"[{request_id}] --- Finished Video Material Generation ---")
        logger.info(f"[{request_id}] Video input options: {video_input_options}")
        logger.info(f"[{request_id}] Video filter inputs: {video_filter_inputs}")
        logger.info(f"[{request_id}] Actual total processed video duration: {actual_total_processed_video_duration} seconds")

        # 3. Narration Generation
        current_step = "narration_generation"
        logger.info(f"[{request_id}] Step 3: Generating narration...")
        if req.narration_enabled and ELEVENLABS_API_KEY:
            narration_plan = parsed_prompt_data.get("narration", {})
            narration_segments_claude = narration_plan.get("segments", [])
            if narration_segments_claude:
                async with httpx.AsyncClient(timeout=ELEVENLABS_API_TIMEOUT) as client_el:
                    for i_nar, seg_info_el in enumerate(narration_segments_claude):
                        nar_summary_entry = {"segment_index": i_nar, "text": seg_info_el.get("text"), "path": None, "error": None, "selected_voice_id": None}
                        segment_text_to_speak = seg_info_el.get("text")
                        if not segment_text_to_speak or not segment_text_to_speak.strip():
                            nar_summary_entry["error"] = "Empty text."
                            generated_files_summary["narration_audios"].append(nar_summary_entry)
                            continue
                        
                        voice_id_to_use = req.narration_voice_id or get_default_voice_id(req.narration_lang)
                        nar_summary_entry["selected_voice_id"] = voice_id_to_use

                        el_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id_to_use}"
                        payload_el = {"text": segment_text_to_speak, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
                        try:
                            response_tts = await client_el.post(el_url, headers={"xi-api-key": ELEVENLABS_API_KEY, "Accept": "audio/mpeg"}, json=payload_el)
                            response_tts.raise_for_status()
                            narration_file_p = temp_dir / f"narration_segment_{i_nar}.mp3"
                            narration_file_p.write_bytes(response_tts.content)
                            nar_summary_entry["path"] = str(narration_file_p.resolve())
                            logger.info(f"[{request_id}] Narration segment {i_nar+1} generated: {narration_file_p.name}")
                        except Exception as e_elevenlabs: 
                            logger.error(f"[{request_id}] ElevenLabs API error for segment {i_nar+1}: {e_elevenlabs}", exc_info=True)
                            nar_summary_entry["error"] = f"ElevenLabs API error: {str(e_elevenlabs)[:100]}"
                        generated_files_summary["narration_audios"].append(nar_summary_entry)
                        if nar_summary_entry.get("error"): generated_files_summary["errors"].append({"step": current_step, "segment_index": i_nar, "error": nar_summary_entry['error']})
                        save_status(task_id, {"status": "processing", "message": f"Generated narration for segment {i_nar+1}/{len(narration_segments_claude)}", "current_step": current_step, "details": generated_files_summary})
                        await asyncio.sleep(0.5) # API rate limiting
            else: logger.info(f"[{request_id}] No narration segments from Claude.")
        elif not req.narration_enabled: logger.info(f"[{request_id}] Narration disabled.")
        else: 
            logger.warning(f"[{request_id}] Narration enabled but ELEVENLABS_API_KEY missing.")
            generated_files_summary["errors"].append({"step": current_step, "error": "ELEVENLABS_API_KEY missing for narration."})
        save_status(task_id, {"status": "processing", "message": "Narration generation step finished.", "current_step": current_step, "details": generated_files_summary})
        
        # 4. Subtitle Generation
        current_step = "subtitle_generation"
        logger.info(f"[{request_id}] Step 4: Generating subtitles...")
        srt_content_str: Optional[str] = None
        if req.subtitles_enabled:
            valid_narration_paths = [
                nar_info["path"] 
                for nar_info in generated_files_summary.get("narration_audios", []) 
                if nar_info.get("path") and Path(nar_info["path"]).exists()
            ]
            
            # Option 1: AssemblyAI transcription
            if ASSEMBLYAI_API_KEY and valid_narration_paths:
                logger.info(f"[{request_id}] Attempting subtitle generation via AssemblyAI from narration.")
                # For now, use the first valid narration audio for transcription.
                # A more robust solution might concatenate narrations before transcription or transcribe each.
                audio_for_transcription_path = valid_narration_paths[0]
                logger.info(f"[{request_id}] Transcribing audio file: {audio_for_transcription_path}")
                async with httpx.AsyncClient(timeout=ASSEMBLYAI_API_TIMEOUT) as client_asm:
                    try:
                        with open(audio_for_transcription_path, "rb") as f:
                            files = {'file': (Path(audio_for_transcription_path).name, f, 'audio/mpeg')}
                            upload_response = await client_asm.post("https://api.assemblyai.com/v2/upload", headers={"authorization": ASSEMBLYAI_API_KEY}, files=files)
                        upload_response.raise_for_status()
                        upload_url_assembly = upload_response.json().get("upload_url")

                        if not upload_url_assembly: raise Exception("AssemblyAI upload failed to return URL.")
                        
                        transcript_payload = {"audio_url": upload_url_assembly, "word_timestamps": True} # Request word timestamps
                        transcript_post_response = await client_asm.post("https://api.assemblyai.com/v2/transcript", headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}, json=transcript_payload)
                        transcript_post_response.raise_for_status()
                        transcript_id_assembly = transcript_post_response.json().get("id")

                        if not transcript_id_assembly: raise Exception("AssemblyAI failed to return transcript ID.")

                        logger.info(f"[{request_id}] AssemblyAI transcription started (ID: {transcript_id_assembly}). Polling...")
                        transcript_get_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id_assembly}"
                        
                        for _ in range(int(ASSEMBLYAI_API_TIMEOUT / ASSEMBLYAI_POLL_INTERVAL)):
                            await asyncio.sleep(ASSEMBLYAI_POLL_INTERVAL)
                            poll_response = await client_asm.get(transcript_get_url, headers=headers_assembly)
                            poll_response.raise_for_status()
                            transcript_result = poll_response.json()
                            status = transcript_result.get("status")
                            logger.info(f"[{request_id}] AssemblyAI poll status: {status}")
                            if status == "completed":
                                generated_files_summary["assembly_ai_raw_transcript"] = transcript_result
                                srt_content_str = await _format_assemblyai_transcript_to_srt(transcript_result)
                                logger.info(f"[{request_id}] AssemblyAI transcription completed and SRT formatted.")
                                break
                            elif status == "error":
                                raise Exception(f"AssemblyAI transcription failed: {transcript_result.get('error')}")
                        else:
                            raise Exception(f"AssemblyAI transcription timed out for ID: {transcript_id_assembly}")
                    except Exception as e_assembly:
                        logger.error(f"[{request_id}] AssemblyAI error: {e_assembly}", exc_info=True)
                        generated_files_summary["errors"].append({"step": current_step, "error": f"AssemblyAI error: {e_assembly}"})
                        srt_content_str = None # Ensure it's None if AssemblyAI fails

            # Option 2: Claude-generated subtitles (if AssemblyAI failed or wasn't used)
            if not srt_content_str and parsed_prompt_data.get("subtitles"):
                logger.info(f"[{request_id}] Using Claude-generated subtitles.")
                srt_content_str = await _format_claude_subtitles_to_srt(
                    parsed_prompt_data["subtitles"],
                    generated_files_summary["scenes"], 
                    actual_total_scene_duration,
                    request_id
                )
                if not srt_content_str:
                    logger.warning(f"[{request_id}] Claude subtitle generation resulted in empty SRT.")
                    generated_files_summary["errors"].append({"step": current_step, "error": "Claude subtitle generation resulted in empty SRT."})

            # Option 3: User-provided script (if other methods failed or weren't used)
            if not srt_content_str and req.subtitle_script_prompt:
                logger.info(f"[{request_id}] Using user-provided subtitle script.")
                duration_for_srt = actual_total_scene_duration if actual_total_scene_duration > 0 else (req.duration or 10.0)
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
                logger.info(f"[{request_id}] No subtitle content generated or enabled.")
        else:
            logger.info(f"[{request_id}] Subtitle generation is disabled by user request.")
        save_status(task_id, {"status": "processing", "message": "Subtitle generation step finished.", "current_step": current_step, "details": generated_files_summary})

        # 5. BGM Generation
        current_step = "bgm_generation"
        logger.info(f"[{request_id}] Step 5: Generating BGM...")
        if req.bgm_enabled:
            if STABILITY_API_KEY:
                bgm_prompt_text = parsed_prompt_data.get("bgm", {}).get("description") or req.bgm_prompt or "calm instrumental background music"
                
                bgm_duration_seconds = int(actual_total_scene_duration) if actual_total_scene_duration > 0 else (req.duration or 10)
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
                                "Accept": "audio/mpeg" 
                            },
                            data=data 
                        )
                        response.raise_for_status()
                        
                        bgm_file_path = temp_dir / f"bgm_audio_{request_id}.mp3"
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
        else:
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
        final_video_output_path = final_output_dir / final_video_filename

        ffmpeg_cmd = [FFMPEG_COMMAND, "-y"]
        ffmpeg_cmd.extend(video_input_options) # Video inputs are already prepared

        audio_input_options: List[str] = []
        audio_filter_inputs: Dict[str, str] = {} 
        current_ffmpeg_input_index = len(video_filter_inputs) # Start counting from after video inputs

        # Narration processing for FFmpeg
        valid_narration_paths = [
            nar_info["path"] 
            for nar_info in generated_files_summary.get("narration_audios", []) 
            if nar_info.get("path") and Path(nar_info["path"]).exists()
        ]

        narration_concat_file_path_str = None
        if req.narration_enabled and valid_narration_paths:
            if len(valid_narration_paths) > 1:
                narration_concat_file_path = temp_dir / f"narrations_concat_{request_id}.txt"
                narration_concat_file_path_str = str(narration_concat_file_path.resolve())
                with open(narration_concat_file_path, "w") as f:
                    for path_str in valid_narration_paths:
                        # Ensure paths are correctly formatted for FFmpeg concat demuxer
                        f.write(f"file '{Path(path_str).as_posix()}'\n")
                audio_input_options.extend(["-f", "concat", "-safe", "0", "-i", narration_concat_file_path_str])
                audio_filter_inputs["narration"] = f"[{current_ffmpeg_input_index}:a]"
                current_ffmpeg_input_index += 1
            elif len(valid_narration_paths) == 1:
                audio_input_options.extend(["-i", str(Path(valid_narration_paths[0]).resolve())])
                audio_filter_inputs["narration"] = f"[{current_ffmpeg_input_index}:a]"
                current_ffmpeg_input_index += 1
            logger.info(f"[{request_id}] Narration inputs for FFmpeg: {audio_input_options}")
            logger.info(f"[{request_id}] Narration filter inputs for FFmpeg: {audio_filter_inputs}")

        # BGM processing for FFmpeg
        bgm_file_path_str = generated_files_summary.get("bgm_audio_file")
        if req.bgm_enabled and bgm_file_path_str and Path(bgm_file_path_str).is_file():
            audio_input_options.extend(["-i", str(Path(bgm_file_path_str).resolve())])
            audio_filter_inputs["bgm"] = f"[{current_ffmpeg_input_index}:a]"
            current_ffmpeg_input_index += 1
            logger.info(f"[{request_id}] BGM input added for FFmpeg: {bgm_file_path_str}")
            logger.info(f"[{request_id}] BGM filter input for FFmpeg: {audio_filter_inputs['bgm']}")

        ffmpeg_cmd.extend(audio_input_options)

        filter_complex_parts = []
        
        # Video chain
        if not video_filter_inputs:
            error_message = "No video inputs available for FFmpeg processing."
            logger.error(f"[{request_id}] {error_message}")
            generated_files_summary["errors"].append({"step": current_step, "error": error_message})
            raise ValueError(error_message)

        video_processing_chain = ""
        if len(video_filter_inputs) > 1:
            concat_inputs = "".join(video_filter_inputs)
            video_processing_chain = f"{concat_inputs}concat=n={len(video_filter_inputs)}:v=1:a=0[vconcat_out];"
            video_stream_for_scaling = "[vconcat_out]"
        else:
            video_stream_for_scaling = video_filter_inputs[0] # e.g., "[0:v]"

        target_w, target_h = parse_resolution(req.resolution)
        video_processing_chain += f"{video_stream_for_scaling}fps={DEFAULT_FPS},format=yuv420p,scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black[vscaled_out]"
        filter_complex_parts.append(video_processing_chain)
        
        final_video_stream_label = "[vscaled_out]"

        # Subtitle filter
        subtitle_file_path = generated_files_summary.get("subtitle_file")
        if req.subtitles_enabled and subtitle_file_path and Path(subtitle_file_path).exists():
            logger.info(f"[{request_id}] Adding subtitles from: {subtitle_file_path} to FFmpeg command.")
            escaped_subtitle_path = str(Path(subtitle_file_path).resolve()).replace('\\', '/').replace(':', '\\\\:') # FFmpeg path escaping
            filter_complex_parts.append(f"{final_video_stream_label}subtitles='{escaped_subtitle_path}'[vout]")
            final_video_map_label = "[vout]"
        else:
            final_video_map_label = final_video_stream_label # No subtitles, use the scaled output directly

        # Audio chain
        narration_stream_ffmpeg = audio_filter_inputs.get("narration")
        bgm_stream_ffmpeg = audio_filter_inputs.get("bgm")
        final_audio_map_label = None

        if narration_stream_ffmpeg and bgm_stream_ffmpeg:
            filter_complex_parts.append(f"{bgm_stream_ffmpeg}volume=0.3[bgm_adjusted]; {narration_stream_ffmpeg}[bgm_adjusted]amix=inputs=2:duration=longest[aout]")
            final_audio_map_label = "[aout]"
        elif narration_stream_ffmpeg:
            filter_complex_parts.append(f"{narration_stream_ffmpeg}acopy[aout]")
            final_audio_map_label = "[aout]"
        elif bgm_stream_ffmpeg:
            filter_complex_parts.append(f"{bgm_stream_ffmpeg}acopy[aout]")
            final_audio_map_label = "[aout]"

        if filter_complex_parts:
            ffmpeg_cmd.extend(["-filter_complex", ";".join(filter_complex_parts)])
        
        ffmpeg_cmd.extend(["-map", final_video_map_label])
        if final_audio_map_label:
            ffmpeg_cmd.extend(["-map", final_audio_map_label])
        else:
            ffmpeg_cmd.append("-an") # No audio

        # Output options
        ffmpeg_cmd.extend([
            "-c:v", "libx264", 
            "-preset", "medium", 
            "-crf", "23", 
            "-c:a", "aac", 
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(final_output_path_temp)
        ])

        logger.info(f"[{request_id}] Final FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        success, ffmpeg_stdout, ffmpeg_stderr = await run_ffmpeg_command_async(ffmpeg_cmd, request_id)
        generated_files_summary["ffmpeg_final_video_log"] = {
            "stdout": ffmpeg_stdout,
            "stderr": ffmpeg_stderr,
            "command": " ".join(ffmpeg_cmd)
        }

        if success:
            generated_files_summary["final_video_path"] = str(final_output_path_temp.resolve())
            
            permanent_video_path = final_output_dir / final_video_filename
            shutil.move(str(final_output_path_temp), str(permanent_video_path))
            final_video_successfully_moved = True
            
            generated_files_summary["final_video_path"] = str(permanent_video_path.resolve())
            generated_files_summary["final_video_url"] = f"/static/generated_videos/{request_id}/{final_video_filename}"
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
            if final_video_successfully_moved or (generated_files_summary["errors"] and not generated_files_summary.get("final_video_path")):
                 logger.info(f"[{request_id}] Cleaning up temporary directory: {temp_dir.resolve()}")
                 shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                 logger.info(f"[{request_id}] Temporary files for video generation kept at {temp_dir.resolve()} as final video might be there or CLEANUP_TEMP_FILES is False.")
            
    return generated_files_summary


@router.post("/generate_from_text/", response_model=VideoGenerationResponse)
async def generate_video_from_text_endpoint(
    req: VideoGenerationRequest, 
    fastapi_request: Request, 
    background_tasks: BackgroundTasks,
    # current_user: models.User = Depends(get_current_active_user) # Ensure models.User is imported
):
    # logger.info(f"User {current_user.email} (ID: {current_user.id}) initiated video generation with prompt: '{req.prompt[:50]}...'")
    logger.info(f"Video generation initiated with prompt: '{req.prompt[:50]}...'") # Simplified logging for now
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

    background_tasks.add_task(create_video_from_text_pipeline, req, fastapi_request, task_id)
    
    return VideoGenerationResponse(
        message="Video generation process started in the background.",
        task_id=task_id,
        status_url=f"{fastapi_request.url_for('get_task_status_endpoint', task_id=task_id)}" 
    )


@router.get("/status/{task_id}", response_model=TaskStatus) 
async def get_task_status_endpoint(task_id: str, request: Request):  # Added Request for URL building
    status_data = get_status_from_file(task_id)
    if not status_data:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found or task not yet started.")
    
    # Ensure all fields expected by TaskStatus are present, providing defaults if necessary
    data_to_return = {
        "task_id": status_data.get("task_id", task_id),
        "status": status_data.get("status", "unknown"),
        "message": status_data.get("message"),
        "progress": status_data.get("progress"),
        "result_url": status_data.get("result_url"), 
        "details": status_data.get("details"),
        "error_details": status_data.get("error_details") 
    }
    # Construct full URL for debug_data_url if summary_file_path exists
    if status_data.get("details") and status_data["details"].get("summary_file_path"):
        summary_file_name = Path(status_data["details"]["summary_file_path"]).name
        # Ensure the path uses the correct task_id for the static route
        data_to_return["debug_data_url"] = str(request.url_for('serve_generated_video_debug_file', task_id=task_id, filename=summary_file_name))


    return TaskStatus(**data_to_return)

# This endpoint will be mounted in main.py
# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")
# This specific one is for the generated videos, assuming they are in a subfolder of static
# It's better to handle this in main.py where the app instance is defined.
# For now, let's assume STATIC_VIDEO_DIR is correctly configured to be served by StaticFiles in main.py

# This is a placeholder for the actual `get_claude_response` and `IndividualAIResponse`
# which should be imported from `main.py` or their respective modules.
# For testing this file standalone, these mocks are useful.
if __name__ != "__main__": # Only define these if not running as main script (i.e., when imported by uvicorn)
    try:
        from main import get_claude_response, IndividualAIResponse, app as main_app
        # from models import User # Assuming User model is in models.py
        # from dependencies import get_current_active_user # Assuming get_current_active_user is in dependencies.py
    except ImportError as e:
        logger.warning(f"Could not import from main/models/dependencies: {e}. Using placeholders.")
        # Fallback definitions already provided above will be used.
        pass
else:
    # If running this file directly, ensure these are defined for local testing if needed.
    REPLICATE_MAX_POLL_ATTEMPTS = 30 
    FFMPEG_COMMAND = "ffmpeg"
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # This is needed for request.url_for to work when running this file directly
    # In a real app, this would be handled by the main FastAPI app instance
    # For testing, we can mount it here.
    # Note: The path "/static/generated_videos" should match STATIC_VIDEO_DIR_BASE
    # and how it's served by the main application.
    if not STATIC_VIDEO_DIR.exists(): # Use STATIC_VIDEO_DIR which is already a Path object
        STATIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Mount the directory where generated videos are stored
    # The path "/static/generated_videos" in url_for will be resolved by this.
    app.mount("/static/generated_videos", StaticFiles(directory=str(STATIC_VIDEO_DIR_BASE)), name="static_videos_generated_direct")


# Separate endpoint for serving debug JSON files
@router.get(f"/static/generated_videos/{{task_id}}/{{filename:path}}")
async def serve_generated_video_debug_file(task_id: str, filename: str):
    file_path = STATIC_VIDEO_DIR / task_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if filename.endswith(".json"):
        return FileResponse(path=str(file_path), media_type="application/json")
    return FileResponse(path=str(file_path))

# The upload_video endpoint and its process_uploaded_video function 
# are temporarily removed to focus on text-to-video generation.
# If needed, they can be restored from previous versions.

```
```python
# Configuration settings for the FastAPI application

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# API Keys
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY") # Corrected variable name
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# API Endpoints and Model Identifiers
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_ZEROSCOPE_MODEL = "anotherjesse/zeroscope-v2-xl:9f7476737190e1a712580adfd5446408f14b2de0e6e8e168d68f2029fc221216"
STABILITY_TEXT_TO_IMAGE_API_URL_BASE = "https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
STABILITY_DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0" # Example, use the latest or preferred engine
STABILITY_AUDIO_API_URL = "https://api.stability.ai/v1/generation/stable-audio-generate-v1" # Corrected endpoint

# Polling and Timeout settings for API calls
REPLICATE_POLL_INTERVAL = 10  # seconds
REPLICATE_API_TIMEOUT = 300.0  # seconds
STABILITY_API_TIMEOUT = 180.0  # seconds, increased for potentially longer audio generation
ELEVENLABS_API_TIMEOUT = 60.0  # seconds
ASSEMBLYAI_POLL_INTERVAL = 5   # seconds
ASSEMBLYAI_API_TIMEOUT = 300.0 # seconds

# Video Processing Defaults
DEFAULT_FPS = 25

# Directory Settings
TEMP_DIR_BASE = "temp_files"  # Base directory for temporary files during processing
STATIC_DIR = "static" # General static directory
STATIC_VIDEO_DIR_NAME = "generated_videos" # Subdirectory for generated videos
STATIC_VIDEO_DIR = Path(STATIC_DIR) / STATIC_VIDEO_DIR_NAME # Full path to static videos directory

# File Management
CLEANUP_TEMP_FILES = True  # Set to False for debugging to keep intermediate files

# Language and Voice Mappings
ELEVENLABS_LANG_VOICE_MAP = {
    "en": "21m00Tcm4TlvDq8ikA2E",  # Default English voice (Rachel)
    "ja": "SOYHLrjzK2X1ezoPC6cr",  # Example Japanese voice
    "es": "0vrPGvXHhDD3rbGURCk8",  # Example Spanish voice
    "fr": "iRYhWuT8tKZ81GesmMsh",  # Example French voice
    "de": "sx7WD8TJIOrk5RQOptDH",  # Example German voice
    "it": "fzDFBB4mgvMlL36gPXcz",      # Italian
    "zh": "4VZIsMPtgggwNg7OXbPY",      # Chinese
    "ko": "WqVy7827vjE2r3jWvbnP",      # Korean
    # Add other languages and corresponding voice IDs as needed
}

DEEPL_LANG_MAP = {
    "en": "EN-US", # DeepL uses EN-US or EN-GB for English
    "ja": "JA",
    "es": "ES",
    "fr": "FR",
    "de": "DE",
    "it": "IT",
    "zh": "ZH",
    # Add more as needed
}

# Other constants
REPLICATE_MAX_POLL_ATTEMPTS = int(REPLICATE_API_TIMEOUT / REPLICATE_POLL_INTERVAL) if REPLICATE_POLL_INTERVAL > 0 else 30
FFMPEG_COMMAND = "ffmpeg" # Ensure ffmpeg is in PATH or provide full path

# Ensure static directories exist
Path(TEMP_DIR_BASE).mkdir(parents=True, exist_ok=True)
STATIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# You can add more configuration variables here as needed
# For example, default model IDs for text-to-image if not using Stability AI's default
# Or settings for quality, style, etc.

# Ensure API keys are loaded (basic check)
if not STABILITY_API_KEY:
    print("Warning: STABILITY_API_KEY is not set in the environment variables.")
if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN is not set in the environment variables.")
if not ELEVENLABS_API_KEY:
    print("Warning: ELEVENLABS_API_KEY is not set in the environment variables.")
if not ASSEMBLYAI_API_KEY:
    print("Warning: ASSEMBLYAI_API_KEY is not set in the environment variables.")
if not DEEPL_API_KEY:
    print("Warning: DEEPL_API_KEY is not set in the environment variables.")

```
