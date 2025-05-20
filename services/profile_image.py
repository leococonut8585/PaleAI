import logging
import shutil
from pathlib import Path
import httpx
import asyncio

from utils.openai_client import openai_client
from utils.sd_client import sd_client, MODEL_ID

logger = logging.getLogger(__name__)


async def _gen_with_dalle(prompt: str) -> bytes:
    print("DALL·E prompt:", prompt)
    params = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
        "quality": "standard",
        "style": "vivid",
    }
    print("DALL·E request params:", params)
    resp = await openai_client.images.generate(**params)
    print("DALL·E raw response:", resp)
    url = resp.data[0].url
    print("DALL·E image URL:", url)
    async with httpx.AsyncClient() as cx:
        content = (await cx.get(url)).content
    return content


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
        return sd_client.run(f"{MODEL_ID}:main", input=params)

    url = await asyncio.to_thread(_run)
    print("SDXL image URL:", url)
    async with httpx.AsyncClient() as cx:
        content = (await cx.get(url)).content
    print("SDXL image bytes:", len(content))
    return content


async def generate_and_save(c1_hex: str, c2_hex: str, gender: str, user_id: int) -> None:
    """Generate a profile image via DALL·E 3 with SDXL fallback and save it."""

    prompt = (
        f"Front-facing cyborg monkey avatar, transparent background. "
        f"Use ONLY these two hex colors {c1_hex} and {c2_hex}. "
        + ("Strong and bold." if gender == 'male' else "Soft and gentle." if gender == 'female' else "Neutral.")
    )
    neg = "more than 2 colors, gradients, photo, text"

    try:
        img_bytes = await _gen_with_dalle(prompt)
    except Exception as e:
        logger.warning(f"DALL·E failed, fallback to SDXL: {e}")
        print("DALL·E error:", e)
        try:
            img_bytes = await _gen_with_sdxl(prompt, neg)
        except Exception as e2:
            logger.error(f"SDXL also failed: {e2}")
            print("SDXL error:", e2)
            Path("static/profile").mkdir(exist_ok=True, parents=True)
            shutil.copy("Pic/デフォルト.png", f"static/profile/{user_id}.png")
            return

    Path("static/profile").mkdir(exist_ok=True, parents=True)
    (Path("static/profile") / f"{user_id}.png").write_bytes(img_bytes)
    print("Saved image for user", user_id)
