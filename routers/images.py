# routers/images.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.concurrency import run_in_threadpool
import os
import base64
import logging
# import httpx  # removed: upscale_image no longer uses it
# import tempfile  # removed: upscale_image no longer uses it

import models
from dependencies import get_current_active_user

router = APIRouter(prefix="/images", tags=["Images"])

logger = logging.getLogger(__name__)

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
        logger.error(f"DeepL error: {e}")
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
            model="claude-opus-4-20250514"
        )
        if res and res.response:
            optimized_text = res.response.strip()
            if optimized_text.lower().startswith("optimized prompt:"):
                optimized_text = optimized_text[len("optimized prompt:"):].strip()
            logger.info(f"Optimized prompt by Claude ({res.source}): {optimized_text}")
            return optimized_text
        else:
            logger.warning(
                f"Claude prompt optimization returned no response. Using original English prompt: {english_prompt}"
            )
            if res and res.error:
                logger.warning(f"Claude optimization error details: {res.error}")
            return english_prompt
    except Exception as e:
        logger.error(f"Claude prompt optimization exception: {e}")
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
            logger.debug("DALL·E request params: %s", params)
            try:
                res = await openai_client.images.generate(**params)
                if res.data and res.data[0].url:
                    logger.info("DALL·E image URL: %s", res.data[0].url)
                    out.append(res.data[0].url)
                else:
                    logger.warning("DALL·E response did not contain a valid URL.")
            except Exception as e_dalle:
                logger.error("DALL·E individual generation failed: %s", e_dalle)
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

        stability_engine_id = "stable-diffusion-3.5"
        logger.info("Using Stability Engine: %s", stability_engine_id)

        stability = stability_client.StabilityInference(key=key, verbose=True, engine=stability_engine_id)

        negative_prompt_text_for_api = ""
        if not req.allow_text:
            negative_prompt_text_for_api = "text, words, letters, watermark, signature, blurry, deformed, ugly, worst quality, low quality, poorly drawn, bad anatomy, extra limbs, missing limbs"

        out: List[str] = []
        for i in range(internal_req_count):
            current_seed = base_seed + req.count + i + 2000

            prompt_list_for_api = [
                generation.Prompt(text=current_optimized_prompt, parameters=generation.PromptParameters(weight=1.0))
            ]
            if negative_prompt_text_for_api:
                prompt_list_for_api.append(
                    generation.Prompt(text=negative_prompt_text_for_api, parameters=generation.PromptParameters(weight=-1.0))
                )

            call_params_for_log = {
                "engine": stability_engine_id,
                "prompts_count": len(prompt_list_for_api),
                "main_prompt_for_log": current_optimized_prompt,
                "negative_prompt_for_log": negative_prompt_text_for_api if negative_prompt_text_for_api else "N/A",
                "seed": current_seed, "steps": 30, "cfg_scale": 7,
                "samples": 1, "width": 1024, "height": 1024
            }
            logger.debug("Stable Diffusion call parameters (for logging): %s", call_params_for_log)

            try:
                answer = await run_in_threadpool(
                    stability.generate,
                    prompt=prompt_list_for_api,
                    seed=current_seed,
                    steps=30,
                    cfg_scale=7.0,
                    width=1024,
                    height=1024,
                    samples=1
                )

                # ☆☆☆ 生のAPI応答をログに出力 ☆☆☆
                logger.debug(
                    "Stable Diffusion API raw answer for seed %s: %s",
                    current_seed,
                    answer,
                )

                found_image = False  # このシードで画像が見つかったかどうかのフラグ
                for resp_idx, resp in enumerate(answer):  # answerがイテラブルであることを期待
                    logger.debug("  Response block %s: %s", resp_idx, type(resp))  # 個々のレスポンスブロックの型
                    for art_idx, art in enumerate(resp.artifacts):  # artifactsがイテラブルであることを期待
                        logger.debug(
                            "    Artifact %s: type=%s, finish_reason=%s",
                            art_idx,
                            art.type,
                            art.finish_reason,
                        )
                        if art.finish_reason == generation.FILTER:
                            logger.warning("    Stable Diffusion image filtered by safety filter.")
                            errors_occurred.append(f"Stable Diffusion (seed {current_seed}): Image filtered by safety.")
                            continue
                        if art.type == generation.ARTIFACT_IMAGE:
                            b64 = base64.b64encode(art.binary).decode()
                            url = f"data:image/png;base64,{b64}"
                            out.append(url)
                            found_image = True
                            logger.debug(
                                "    Successfully processed artifact image for seed %s",
                                current_seed,
                            )

                if not found_image:
                    logger.warning(
                        "  No valid image artifact found in API response for seed %s.",
                        current_seed,
                    )
                    # errors_occurred に何も追加しないでおくと、単に画像が0枚だったことになる
                    # エラーとして扱うべきか検討 ( 例: APIは成功したが期待した画像がなかった場合 )

            except Exception as e_stable_gen:
                logger.error(
                    "Stable Diffusion individual generation failed for seed %s: %s",
                    current_seed,
                    e_stable_gen,
                )
                errors_occurred.append(f"Stable Diffusion Exception (seed {current_seed}): {str(e_stable_gen)}")
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
            logger.info(
                "Attempting OpenAI (DALL·E) generation for %s image(s)...",
                openai_images_to_generate,
            )
            openai_urls = await dalle_try(openai_images_to_generate, optimized)
            all_generated_urls.extend(openai_urls)
        except Exception as e:
            logger.error("OpenAI (DALL·E) main call failed: %s", e)
            errors_occurred.append(f"OpenAI Main Error: {str(e)}")

    if stable_images_to_generate > 0:
        stable_optimized_prompt = optimized
        try:
            logger.info(
                "Attempting Stable Diffusion generation for %s image(s)...",
                stable_images_to_generate,
            )
            stable_urls = await stable_try(stable_images_to_generate, stable_optimized_prompt)
            all_generated_urls.extend(stable_urls)
        except Exception as e:
            logger.error("Stable Diffusion main call failed: %s", e)
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

