from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.concurrency import run_in_threadpool
import os
import base64
import httpx
import tempfile

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
    replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        print("Replicate API token not found. Skipping upscale.")
        return url

    import replicate

    tmp_file_path = None
    try:
        print(f"Original image URL for upscale: {url}")
        async with httpx.AsyncClient() as client_http:
            response = await client_http.get(url, timeout=30.0)
            response.raise_for_status()
            image_bytes = response.content

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_file_path = tmp_file.name

        print(f"Image downloaded to temporary file: {tmp_file_path}")

        input_payload = {"image": open(tmp_file_path, "rb")}

        print(f"Calling Replicate API with temporary file: {tmp_file_path}")
        output = await run_in_threadpool(
            replicate.run,
            "cjwbw/real-esrgan:e2ec5874a9427a78cb24f52b8798dfc778e7f412378e5f1fcd4730aa0586456b",
            input=input_payload
        )

        input_payload["image"].close()

        print(f"Replicate upscale raw response: {output}")

        if isinstance(output, str):
            result_url = output
        elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], str):
            result_url = output[0]
        else:
            print(f"Unexpected Replicate output format: {type(output)}. Using original URL.")
            if tmp_file_path:
                os.remove(tmp_file_path)
            return url

        print(f"Replicate upscale result URL: {result_url}")
        if tmp_file_path:
            os.remove(tmp_file_path)
        return result_url

    except httpx.HTTPStatusError as e_http:
        print(f"Failed to download image from DALL·E URL: {e_http}. Response: {e_http.response.text}")
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        return url
    except replicate.exceptions.ReplicateError as e_replicate:
        print(f"Replicate API error: {e_replicate}")
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        return url
    except Exception as e:
        print(f"General upscale error: {e}")
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        return url
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                input_payload["image"].close()
            except NameError:
                pass
            except AttributeError:
                pass
            try:
                os.remove(tmp_file_path)
                print(f"Temporary file {tmp_file_path} removed in finally block.")
            except OSError as e_os:
                print(f"Error removing temporary file {tmp_file_path} in finally block: {e_os}")

@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(req: ImageGenerationRequest, current_user: models.User = Depends(get_current_active_user)):
    english = await translate_prompt(req.prompt)
    if req.text_content:
        english += f" text:{req.text_content}"
    if not req.allow_text:
        english += " no letters, watermark, text"
    optimized = await optimize_prompt(english)
    all_generated_urls: List[str] = []
    errors_occurred: List[str] = []

    import random
    base_seed = random.randint(0, 2**32 - 1)
    cfg_scale = 5 + req.deference

    async def dalle_try(internal_req_count: int) -> List[str]:
        from utils.openai_client import openai_client
        if not openai_client:
            raise Exception("OpenAI is not configured")
        out: List[str] = []
        for i in range(internal_req_count):
            current_seed = base_seed + i
            params = {
                "model": "dall-e-3",
                "prompt": optimized,
                "n": 1,
                "size": "1024x1024",
            }
            print("DALL·E request params:", params)
            try:
                res = await openai_client.images.generate(**params)
                print("DALL·E raw response:", res)
                if res.data and res.data[0].url:
                    url = res.data[0].url
                    print("DALL·E image URL:", url)
                    out.append(url)
                else:
                    print("DALL·E response does not contain a valid URL.")
            except Exception as e_dalle:
                print(f"DALL·E individual generation failed: {e_dalle}")
        return out

    async def stable_try(internal_req_count: int) -> List[str]:
        try:
            from stability_sdk import client as stability_client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
        except Exception as e:
            # このエラーはSDKがインストールされていないなど、致命的な場合
            raise Exception(f"Stability SDK import error or critical setup issue: {e}")

        key = os.getenv("STABILITY_API_KEY")
        if not key:
            raise Exception("Stability API key missing. Cannot use Stable Diffusion.")

        stability = stability_client.StabilityInference(key=key, verbose=False, engine="stable-diffusion-xl-1024-v1-0")

        negative_text_for_prompt = "text, watermark, letters, logo, words, typo, signature, blurry, ugly, deformed"

        out: List[str] = []

        for i in range(internal_req_count):
            current_seed = base_seed + req.count + i

            prompts = [generation.Prompt(text=optimized, parameters=generation.PromptParameters(weight=1.0))]
            if not req.allow_text and negative_text_for_prompt:
                prompts.append(generation.Prompt(text=negative_text_for_prompt, parameters=generation.PromptParameters(weight=-1.0)))

            generation_params = {
                "prompt": prompts,
                "steps": 30,
                "seed": current_seed,
                "cfg_scale": cfg_scale,
                "samples": 1,
                "width": 1024,
                "height": 1024,
            }

            print("Stable Diffusion request params (for generate call):", generation_params)
            try:
                answer = await run_in_threadpool(
                    stability.generate,
                    prompt=generation_params["prompt"],
                    seed=generation_params["seed"],
                    steps=generation_params["steps"],
                    cfg_scale=generation_params["cfg_scale"],
                    width=generation_params["width"],
                    height=generation_params["height"],
                    samples=generation_params["samples"]
                )
                for resp in answer:
                    for art in resp.artifacts:
                        if art.finish_reason == generation.FILTER:
                            print("Stable Diffusion image filtered.")
                            continue
                        if art.type == generation.ARTIFACT_IMAGE:
                            b64 = base64.b64encode(art.binary).decode()
                            url = f"data:image/png;base64,{b64}"
                            out.append(url)
            except Exception as e_stable_gen:
                print(f"Stable Diffusion individual generation failed: {e_stable_gen}")
        return out

    requested_api = req.api.lower() if req.api else "openai"
    openai_count = 0
    stable_count = 0

    if requested_api == "openai":
        openai_count = req.count
    elif requested_api == "stable":
        stable_count = req.count
    elif requested_api == "both":
        openai_count = (req.count + 1) // 2
        stable_count = req.count // 2
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported API type: {req.api}")

    if openai_count > 0:
        try:
            print(f"Attempting OpenAI generation for {openai_count} image(s)...")
            openai_urls = await dalle_try(openai_count)
            all_generated_urls.extend(openai_urls)
        except Exception as e:
            print(f"OpenAI (DALL·E) generation failed entirely: {e}")
            errors_occurred.append(f"OpenAI Error: {str(e)}")

    if stable_count > 0:
        try:
            print(f"Attempting Stable Diffusion generation for {stable_count} image(s)...")
            stable_urls = await stable_try(stable_count)
            all_generated_urls.extend(stable_urls)
        except Exception as e:
            print(f"Stable Diffusion generation failed entirely: {e}")
            errors_occurred.append(f"Stable Diffusion Error: {str(e)}")

    if not all_generated_urls:
        error_detail = "Image generation failed for all requested APIs."
        if errors_occurred:
            error_detail += " Errors: " + "; ".join(errors_occurred)
        raise HTTPException(status_code=500, detail=error_detail)

    upscaled_urls: List[str] = []
    if all_generated_urls:
        for u_idx, u_url in enumerate(all_generated_urls):
            print(f"Upscaling image {u_idx+1}/{len(all_generated_urls)}: {u_url[:60]}...")
            try:
                upscaled_url = await upscale_image(u_url)
                upscaled_urls.append(upscaled_url)
            except Exception as e_upscale:
                print(f"Failed to upscale image {u_url[:60]}... Error: {e_upscale}. Using original URL.")
                upscaled_urls.append(u_url)

    final_response_data = {"urls": upscaled_urls}
    if errors_occurred and not upscaled_urls:
        final_response_data["error"] = "; ".join(errors_occurred)
    elif errors_occurred:
        final_response_data["error"] = "Some image generations may have failed. Errors: " + "; ".join(errors_occurred)

    return final_response_data
