from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.concurrency import run_in_threadpool
import os
import base64
# import httpx  # removed: upscale_image no longer uses it
# import tempfile  # removed: upscale_image no longer uses it

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

async def optimize_prompt(english_prompt: str) -> str:
    from main import get_claude_response

    optimization_system_prompt = """You are an expert image prompt engineer.
Your task is to take the user-provided English prompt and refine it for optimal results with text-to-image generation APIs like DALL-E 3 or Stable Diffusion.
Focus on clarity, vivid details, artistic style if implied, and overall coherence.
Preserve the core intent of the original prompt.
**You MUST return ONLY the optimized English prompt and nothing else. No conversational openers, no closers, no explanations, no apologies, no persona, no markdown formatting, no code blocks.**
For example, if the input is "a cat", a good output might be "A photorealistic image of a fluffy ginger cat napping in a sunbeam, soft lighting, detailed fur."
The prompt you are optimizing is in English. Ensure your output is also a single, clean English prompt string.
"""

    user_prompt_for_claude = f"Please optimize the following image generation prompt: \"{english_prompt}\""

    try:
        res = await get_claude_response(
            prompt_text=user_prompt_for_claude,
            system_instruction=optimization_system_prompt,
            model="claude-3-haiku-20240307"
        )
        if res and res.response:
            optimized_text = res.response.strip()
            if optimized_text.lower().startswith("optimized prompt:"):
                optimized_text = optimized_text[len("optimized prompt:"):].strip()
            print(f"Optimized prompt by Claude ({res.source}): {optimized_text}")
            return optimized_text
        else:
            print(f"Claude prompt optimization returned no response. Using original English prompt: {english_prompt}")
            if res and res.error:
                print(f"Claude optimization error details: {res.error}")
            return english_prompt
    except Exception as e:
        print(f"Claude prompt optimization exception: {e}")
        return english_prompt


@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(req: ImageGenerationRequest, current_user: models.User = Depends(get_current_active_user)):
    english_prompt_for_translation = req.prompt
    if req.text_content:
        english_prompt_for_translation += f" with text: \"{req.text_content}\""

    english = await translate_prompt(english_prompt_for_translation)
    optimized = await optimize_prompt(english)

    all_generated_urls: List[str] = []
    errors_occurred: List[str] = []

    import random
    base_seed = random.randint(0, 2**32 - 1)
    cfg_scale = 5 + req.deference

    async def dalle_try(internal_req_count: int, current_optimized_prompt: str) -> List[str]:
        from utils.openai_client import openai_client
        if not openai_client:
            raise Exception("OpenAI client is not configured")
        out: List[str] = []
        for i in range(internal_req_count):
            params = {
                "model": "dall-e-3",
                "prompt": current_optimized_prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "standard",
            }
            print("DALL·E request params:", params)
            try:
                res = await openai_client.images.generate(**params)
                if res.data and res.data[0].url:
                    print(f"DALL·E image URL: {res.data[0].url}")
                    out.append(res.data[0].url)
                else:
                    print("DALL·E response did not contain a valid URL.")
            except Exception as e_dalle:
                print(f"DALL·E individual generation failed: {e_dalle}")
                errors_occurred.append(f"DALL·E: {str(e_dalle)}")
        return out

    async def stable_try(internal_req_count: int, current_optimized_prompt: str) -> List[str]:
        try:
            from stability_sdk import client as stability_client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
        except Exception as e:
            raise Exception(f"Stability SDK import error or critical setup issue: {e}")

        key = os.getenv("STABILITY_API_KEY")
        if not key:
            raise Exception("Stability API key missing. Cannot use Stable Diffusion.")

        stability_engine_id = "stable-diffusion-xl-1024-v1-0"
        print(f"Using Stability Engine: {stability_engine_id}")
        stability = stability_client.StabilityInference(key=key, verbose=False, engine=stability_engine_id)

        negative_prompt_text_base = "blurry, ugly, deformed, worst quality, low quality, poorly drawn, bad anatomy, extra limbs, missing limbs"
        if not req.allow_text:
            negative_prompt_text = negative_prompt_text_base + ", text, watermark, letters, logo, words, typo, signature"
        else:
            negative_prompt_text = negative_prompt_text_base

        out: List[str] = []
        for i in range(internal_req_count):
            current_seed = base_seed + req.count + i + 1000

            prompt_list_for_api = [
                generation.Prompt(text=current_optimized_prompt, parameters=generation.PromptParameters(weight=1.0))
            ]
            if negative_prompt_text:
                prompt_list_for_api.append(
                    generation.Prompt(text=negative_prompt_text, parameters=generation.PromptParameters(weight=-1.0))
                )

            call_params_for_log = {
                "prompt_text_for_log": current_optimized_prompt,
                "negative_prompt_text_for_log": negative_prompt_text if negative_prompt_text else "N/A",
                "seed": current_seed,
                "steps": 30,
                "cfg_scale": cfg_scale,
                "samples": 1,
                "width": 1024,
                "height": 1024,
                "engine": stability_engine_id,
            }
            print("Stable Diffusion call parameters (for logging):", call_params_for_log)

            try:
                answer = await run_in_threadpool(
                    stability.generate,
                    prompt=prompt_list_for_api,
                    seed=current_seed,
                    steps=30,
                    cfg_scale=cfg_scale,
                    width=1024,
                    height=1024,
                    samples=1,
                )
                for resp in answer:
                    for art in resp.artifacts:
                        if art.finish_reason == generation.FILTER:
                            print("Stable Diffusion image filtered by safety filter.")
                            errors_occurred.append("Stable Diffusion: Image filtered by safety.")
                            continue
                        if art.type == generation.ARTIFACT_IMAGE:
                            b64 = base64.b64encode(art.binary).decode()
                            url = f"data:image/png;base64,{b64}"
                            out.append(url)
            except Exception as e_stable_gen:
                print(f"Stable Diffusion individual generation failed for seed {current_seed}: {e_stable_gen}")
                errors_occurred.append(f"Stable Diffusion: {str(e_stable_gen)}")
        return out

    requested_api_choice = req.api.lower() if req.api else "openai"

    openai_images_to_generate = 0
    stable_images_to_generate = 0

    if requested_api_choice == "openai":
        openai_images_to_generate = req.count
    elif requested_api_choice == "stable":
        stable_images_to_generate = req.count
    elif requested_api_choice == "both":
        openai_images_to_generate = (req.count + 1) // 2
        stable_images_to_generate = req.count // 2
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported API choice: {req.api}")

    if openai_images_to_generate > 0:
        try:
            print(f"Attempting OpenAI (DALL·E) generation for {openai_images_to_generate} image(s)...")
            openai_urls = await dalle_try(openai_images_to_generate, optimized)
            all_generated_urls.extend(openai_urls)
        except Exception as e:
            print(f"OpenAI (DALL·E) main call failed: {e}")
            errors_occurred.append(f"OpenAI Main Error: {str(e)}")

    if stable_images_to_generate > 0:
        stable_optimized_prompt = optimized
        try:
            print(f"Attempting Stable Diffusion generation for {stable_images_to_generate} image(s)...")
            stable_urls = await stable_try(stable_images_to_generate, stable_optimized_prompt)
            all_generated_urls.extend(stable_urls)
        except Exception as e:
            print(f"Stable Diffusion main call failed: {e}")
            errors_occurred.append(f"Stable Diffusion Main Error: {str(e)}")

    if not all_generated_urls:
        error_detail_msg = "Image generation failed for all selected APIs."
        if errors_occurred:
            error_detail_msg += " Specific errors: " + "; ".join(errors_occurred)
        raise HTTPException(status_code=500, detail=error_detail_msg)

    response_payload = {"urls": all_generated_urls}
    if errors_occurred:
        response_payload["error"] = "Image generation process completed with some errors: " + "; ".join(errors_occurred)

    return response_payload

