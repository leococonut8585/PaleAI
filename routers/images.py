from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.concurrency import run_in_threadpool
import os
import base64

import models
from dependencies import get_current_active_user

router = APIRouter(prefix="/images", tags=["Images"])

class ImageGenerationRequest(BaseModel):
    prompt: str
    count: int = Field(1, ge=1, le=10)
    api: Optional[str] = "openai"
    # Deference controls variation strength 1(min) - 5(max)
    deference: int = Field(3, ge=1, le=5)
    # When False the generated image must not contain any text
    allow_text: bool = True
    # Explicit text to be drawn in the image if provided
    text_content: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    urls: List[str]
    error: Optional[str] = None

async def translate_prompt(text: str) -> str:
    from main import deepl_translator
    if not deepl_translator:
        return text
    try:
        result = await run_in_threadpool(deepl_translator.translate_text, text, target_lang="EN-US")
        return result.text
    except Exception as e:
        print(f"DeepL error: {e}")
        return text

async def optimize_prompt(text: str) -> str:
    from main import get_claude_response
    try:
        res = await get_claude_response(text)
        if res and res.response:
            return res.response
        return text
    except Exception as e:
        print(f"Claude error: {e}")
        return text

async def upscale_image(url: str) -> str:
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        return url
    import replicate
    client = replicate.Client(api_token=token)
    try:
        output = await run_in_threadpool(client.run, "cjwbw/real-esrgan", input={"image": url})
        if isinstance(output, list):
            return output[-1]
        return str(output)
    except Exception as e:
        print(f"Upscale error: {e}")
        return url

@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(req: ImageGenerationRequest, current_user: models.User = Depends(get_current_active_user)):
    english = await translate_prompt(req.prompt)
    if req.text_content:
        english += f" text:{req.text_content}"
    if not req.allow_text:
        english += " no letters, watermark, text"
    optimized = await optimize_prompt(english)
    urls: List[str] = []
    import random
    base_seed = random.randint(0, 2**32 - 1)
    cfg_scale = 5 + req.deference
    if req.api == "openai":
        from utils.openai_client import openai_client
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI is not configured")
        try:
            for i in range(req.count):
                seed = base_seed if req.deference == 1 else base_seed + i
                res = await openai_client.images.generate(model="dall-e-3", prompt=optimized, n=1, size="1024x1024", seed=seed)
                urls.append(res.data[0].url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI image error: {e}")
    else:
        try:
            from stability_sdk import client as stability_client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stability SDK import error: {e}")
        key = os.getenv("STABILITY_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="Stability API key missing")
        stability = stability_client.StabilityInference(key=key, verbose=False)
        negative = None if req.allow_text else "text, watermark, letters, logo"
        for i in range(req.count):
            seed = base_seed if req.deference == 1 else base_seed + i
            answer = await run_in_threadpool(
                stability.generate,
                prompt=optimized,
                steps=30,
                seed=seed,
                cfg_scale=cfg_scale,
                negative_prompt=negative,
            )
            for resp in answer:
                for art in resp.artifacts:
                    if art.finish_reason == generation.FILTER:
                        continue
                    if art.type == generation.ARTIFACT_IMAGE:
                        b64 = base64.b64encode(art.binary).decode()
                        urls.append(f"data:image/png;base64,{b64}")
    upscaled = []
    for u in urls:
        upscaled.append(await upscale_image(u))
    return {"urls": upscaled}
