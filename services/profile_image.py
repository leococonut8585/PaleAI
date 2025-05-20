import logging
import shutil
from pathlib import Path
import httpx
import asyncio

from utils.openai_client import openai_client
from utils.sd_client import sd_client, MODEL_ID

# Mapping of common hex color codes to English color names
_COLOR_MAP = {
    "#ff0000": "red",
    "#0000ff": "blue",
    "#00ff00": "green",
    "#ffff00": "yellow",
    "#ff00ff": "magenta",
    "#00ffff": "cyan",
    "#000000": "black",
    "#ffffff": "white",
    "#ffa500": "orange",
    "#800080": "purple",
}


def _hex_to_color_name(hex_code: str) -> str:
    """Return a simple English color name for a hex code."""
    return _COLOR_MAP.get(hex_code.lower(), "a custom color")

logger = logging.getLogger(__name__)


async def _gen_with_dalle(prompt: str) -> bytes:
    # Log the actual prompt used for this API call
    print("実際に使ったプロンプト:", prompt)
    params = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
        "quality": "standard",
        "style": "vivid",
    }
    print("DALL·E request params:", params)
    try:
        resp = await openai_client.images.generate(**params)
        print("APIレスポンスやエラー:", resp)
        print("DALL·E raw response:", resp)
        url = resp.data[0].url
        print("DALL·E image URL:", url)
        async with httpx.AsyncClient() as cx:
            content = (await cx.get(url)).content
        return content
    except Exception as e:
        # Log the error before re-raising so retries can inspect it
        print("APIレスポンスやエラー:", e)
        raise


async def _gen_with_dalle_retry(prompts: list[str]) -> bytes:
    """Try multiple prompts in order, softening the wording on invalid request errors."""
    last_error: Exception | None = None
    for i, p in enumerate(prompts):
        try:
            return await _gen_with_dalle(p)
        except Exception as e:
            last_error = e
            code = getattr(e, "code", "")
            status = getattr(e, "status_code", None)
            if i < len(prompts) - 1 and (
                code == "invalid_request_error" or status == 400
            ):
                # Soften the prompt and try again
                continue
            raise


async def _gen_with_sdxl(prompt: str, neg: str) -> bytes:
    print("SDXL prompt:", prompt)
    print("SDXL negative_prompt:", neg)
    def _run():
        params = {
            "prompt": prompt,
            "negative_prompt": neg,
            "width": 1024,
            "height": 1024,
        }
        print("SDXL request params:", params)
        # Pass the SDXL model ID directly without a tag like :main
        return sd_client.run(MODEL_ID, input=params)

    url = await asyncio.to_thread(_run)
    print("SDXL image URL:", url)
    async with httpx.AsyncClient() as cx:
        content = (await cx.get(url)).content
    print("SDXL image bytes:", len(content))
    return content


async def generate_and_save(c1_hex: str, c2_hex: str, gender: str, user_id: int) -> None:
    """Generate a profile image via DALL·E 3 with SDXL fallback and save it."""

    c1_name = _hex_to_color_name(c1_hex)
    c2_name = _hex_to_color_name(c2_hex)
    gender_desc = (
        "With a strong and confident expression." if gender == "男性" else
        "With a gentle and friendly expression." if gender == "女性" else
        "With a neutral and approachable expression."
    )

    def _build_prompt(subject: str) -> str:
        return (
            f"A front-facing {subject}, digital illustration, transparent background. "
            f"Vivid {c1_name} and vivid {c2_name} color scheme. "
            f"{gender_desc} "
            "Cute, friendly, playful, soft colors, simple illustration, no text, no watermark, "
            "no bars, no extra design elements, no shapes, only monkey."
        )

    prompt_variants = [
        _build_prompt("cyborg monkey mascot"),
        _build_prompt("monkey mascot"),
        _build_prompt("monkey character"),
        _build_prompt("animal character"),
    ]
    neg = "more than 2 colors, gradients, photo, text"

    try:
        img_bytes = await _gen_with_dalle_retry(prompt_variants)
    except Exception as e:
        logger.warning(f"DALL·E failed, fallback to SDXL: {e}")
        print("DALL·E error:", e)
        try:
            img_bytes = await _gen_with_sdxl(prompt_variants[0], neg)
        except Exception as e2:
            logger.error(f"SDXL also failed: {e2}")
            print("SDXL error:", e2)
            Path("static/profile").mkdir(exist_ok=True, parents=True)
            shutil.copy("Pic/デフォルト.png", f"static/profile/{user_id}.png")
            return

    Path("static/profile").mkdir(exist_ok=True, parents=True)
    (Path("static/profile") / f"{user_id}.png").write_bytes(img_bytes)
    print("Saved image for user", user_id)
