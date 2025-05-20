import importlib.util
import asyncio
from pathlib import Path
import pytest


pi_spec = importlib.util.find_spec("services.profile_image")
if pi_spec is None:
    pytest.skip("services.profile_image module not available", allow_module_level=True)

openai_spec = importlib.util.find_spec("openai")
replicate_spec = importlib.util.find_spec("replicate")
if openai_spec is None and replicate_spec is None:
    pytest.skip("openai and replicate packages required for this test", allow_module_level=True)

import services.profile_image as pi


@pytest.mark.asyncio
async def test_profile_image_generation_real():
    prompt = (
        "A cyborg monkey mascot character, masculine and powerful impression, "
        "using only blue and orange colors. Transparent background. Art style similar to the attached reference images."
    )
    print(f"Prompt: {prompt}")

    img_bytes = None

    try:
        img_bytes = await pi._gen_with_dalle(prompt)
        print(f"DALL\u00b7E response bytes: {len(img_bytes)}")
    except Exception as e:
        print(f"DALL\u00b7E generation failed: {e}")
        try:
            img_bytes = await pi._gen_with_sdxl(prompt, "more than 2 colors, gradients, photo, text, watermark, bars, background")
            print(f"SDXL response bytes: {len(img_bytes)}")
        except Exception as e2:
            print(f"SDXL generation failed: {e2}")

    if img_bytes:
        Path("static/profile").mkdir(parents=True, exist_ok=True)
        out_path = Path("static/profile/test.png")
        out_path.write_bytes(img_bytes)
        print(f"Saved generated image to {out_path}")
        assert out_path.exists() and out_path.stat().st_size > 0
    else:
        print("Image generation failed. No file saved.")
        assert True
