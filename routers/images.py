# routers/images.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.concurrency import run_in_threadpool
import os
import base64
import logging
# import httpx  # removed: upscale_image no longer uses it
# import tempfile  # removed: upscale_image no longer uses it
import asyncio # Added for Stable Diffusion (Replicate)
import io # Added for DALL-E 2 image editing

import models
from dependencies import get_current_active_user
from utils.openai_client import openai_client # Added for DALL-E 3
from utils.sd_client import sd_client, MODEL_ID # Added for Stable Diffusion (Replicate)

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

# New Pydantic Models
class ImageSettings(BaseModel):
    quality: Optional[str] = "standard"
    size: Optional[str] = "1024x1024"
    format: Optional[str] = "png"
    style: Optional[str] = None
    # Add other common settings if obvious, or leave for later refinement

class ImageGenerationRequestPayload(BaseModel):
    generation_mode: str = Field(..., examples=["high_quality", "speedy", "arrange_image", "generate_from_image"])
    prompt: str
    negative_prompt: Optional[str] = None
    image_settings: ImageSettings = Field(default_factory=ImageSettings) # Use default_factory for nested models
    batch_count: int = 1 # Default to 1, will be used in Phase 2

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

        stability_engine_id = "stable-diffusion-3-medium"
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


@router.post("/generate_new", summary="Generate image with new multi-modal generation pipeline")
async def generate_new_image(
    payload: ImageGenerationRequestPayload = Form(...),
    image_file: Optional[UploadFile] = File(None)
    # current_user: models.User = Depends(get_current_active_user) # Placeholder for auth
):
    # Placeholder for user authentication/authorization if needed
    # For example: if not current_user: raise HTTPException(status_code=401, detail="Not authenticated")

    # Log received data for debugging (optional, can be removed later)
    # logger.debug(f"Received payload: {payload.model_dump()}")
    # if image_file:
    #     logger.debug(f"Received image_file: {image_file.filename}")

    if payload.generation_mode == "high_quality":
        logger.info("Processing 'high_quality' image generation request.")
        
        # Parameter Validation and Preparation
        valid_sizes_dalle3 = ["1024x1024", "1024x1792", "1792x1024"]
        size = payload.image_settings.size if payload.image_settings.size in valid_sizes_dalle3 else "1024x1024"
        
        valid_qualities = ["standard", "hd"]
        quality = payload.image_settings.quality if payload.image_settings.quality in valid_qualities else "standard"
        
        valid_styles = ["vivid", "natural"]
        style = payload.image_settings.style if payload.image_settings.style in valid_styles else "vivid"

        api_params = {
            "model": "dall-e-3",
            "prompt": payload.prompt,
            "n": 1, # For Phase 1, batch_count is not yet used for multiple 'n' values
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": "url",
        }
        logger.debug(f"DALL·E 3 API parameters: {api_params}")

        try:
            response = await openai_client.images.generate(**api_params)
            
            if not response.data or not response.data[0].url:
                logger.error("DALL·E 3 response did not contain expected data.")
                raise HTTPException(
                    status_code=500,
                    detail={"type": "api_error", "message": "DALL·E 3 response missing data."}
                )
            
            image_url = response.data[0].url
            logger.info(f"DALL·E 3 image generated successfully: {image_url}")
            
            return {
                "success": True,
                "images": [{"url": image_url, "prompt_used": payload.prompt}],
                "metadata": {
                    "mode_used": "high_quality",
                    "settings_applied": api_params 
                }
            }

        except HTTPException as http_exc: # Re-raise HTTPExceptions directly
            raise http_exc
        except Exception as e:
            logger.error(f"DALL·E 3 API call failed: {e}", exc_info=True)
            error_details = {"error_type": type(e).__name__, "message": str(e)}
            if hasattr(e, 'body'): # For OpenAI specific errors
                error_details['body'] = e.body
            
            raise HTTPException(
                status_code=500,
                detail={"type": "api_error", "message": "Failed to generate image with DALL·E 3.", "details": error_details}
            )

    elif payload.generation_mode == "speedy":
        logger.info("Processing 'speedy' (Stable Diffusion/Replicate) image generation request.")

        if "REPLACE_WITH_ACTUAL_HASH" in MODEL_ID:
            logger.error("Stable Diffusion MODEL_ID is not configured in utils/sd_client.py.")
            raise HTTPException(
                status_code=500,
                detail={"type": "configuration_error", "message": "Stable Diffusion model not configured."}
            )

        width, height = 1024, 1024 # Default size
        if payload.image_settings.size:
            try:
                w_str, h_str = payload.image_settings.size.split('x')
                parsed_w, parsed_h = int(w_str), int(h_str)
                # Add validation for allowed SD sizes if necessary, or let Replicate handle it
                width, height = parsed_w, parsed_h
            except ValueError:
                logger.warning(f"Invalid size format '{payload.image_settings.size}', using default {width}x{height}.")
        
        api_input = {
            "prompt": payload.prompt,
            "width": width,
            "height": height,
        }
        if payload.negative_prompt:
            api_input["negative_prompt"] = payload.negative_prompt
        
        # Placeholder for other SD-specific settings from payload.image_settings if added later
        # e.g., api_input["num_inference_steps"] = payload.image_settings.steps or 50
        # e.g., api_input["guidance_scale"] = payload.image_settings.cfg_scale or 7.5

        logger.debug(f"Stable Diffusion (Replicate) API input: {api_input}")

        try:
            output_urls = await asyncio.to_thread(sd_client.run, MODEL_ID, input=api_input)
            
            if not output_urls or not isinstance(output_urls, list) or not output_urls[0]:
                logger.error("Stable Diffusion (Replicate) response did not contain expected image URLs.")
                raise HTTPException(
                    status_code=500,
                    detail={"type": "api_error", "message": "Stable Diffusion (Replicate) did not return image URLs."}
                )
            
            logger.info(f"Stable Diffusion (Replicate) images generated successfully: {output_urls}")
            
            images_data = [{"url": url, "prompt_used": payload.prompt} for url in output_urls]

            return {
                "success": True,
                "images": images_data, # Assuming output_urls is a list of URLs
                "metadata": {
                    "mode_used": "speedy",
                    "settings_applied": {
                        "model_id": MODEL_ID,
                        "prompt": payload.prompt,
                        "negative_prompt": payload.negative_prompt,
                        "width": width,
                        "height": height,
                        # Add other applied settings here
                    }
                }
            }
        except HTTPException as http_exc: # Re-raise HTTPExceptions directly
            raise http_exc
        except Exception as e:
            logger.error(f"Stable Diffusion (Replicate) API call failed: {e}", exc_info=True)
            error_details = {"error_type": type(e).__name__, "message": str(e)}
            # Replicate client errors sometimes have more details in e.args or other attributes
            if hasattr(e, 'response') and e.response is not None: # type: ignore
                 error_details['replicate_response_status'] = e.response.status_code # type: ignore
                 try:
                     error_details['replicate_response_body'] = e.response.json() # type: ignore
                 except:
                     error_details['replicate_response_body'] = e.response.text # type: ignore

            raise HTTPException(
                status_code=500,
                detail={"type": "api_error", "message": "Failed to generate image with Stable Diffusion (Replicate).", "details": error_details}
            )

    elif payload.generation_mode == "arrange_image":
        logger.info("Processing 'arrange_image' (DALL·E 2 Edit) request.")
        if not image_file: # This check is technically redundant due to the one below, but good for clarity
            raise HTTPException(status_code=400, detail="Image file is required for 'arrange_image' mode.")

        try:
            image_bytes = await image_file.read()
            image_file_like = io.BytesIO(image_bytes)
            
            # DALL·E 2 edit supports "256x256", "512x512", "1024x1024"
            valid_sizes_dalle2 = ["256x256", "512x512", "1024x1024"]
            dalle2_size = payload.image_settings.size if payload.image_settings.size in valid_sizes_dalle2 else "1024x1024"
            
            # No specific quality or style for DALL-E 2 edit, only size and prompt
            api_params = {
                "prompt": payload.prompt,
                "n": 1, # For Phase 1, batch_count is not yet used for multiple 'n' values
                "size": dalle2_size,
                "response_format": "url",
            }
            logger.debug(f"DALL·E 2 Edit API parameters: {api_params}")

            # Pass the image as a tuple: (filename, file_object)
            # The filename helps the API determine the type, or use content_type if available and needed
            response = await openai_client.images.edit(
                image=(image_file.filename, image_file_like), # Pass as tuple
                prompt=api_params["prompt"],
                n=api_params["n"],
                size=api_params["size"],
                response_format=api_params.get("response_format", "url") # type: ignore
            )

            if not response.data or not response.data[0].url:
                logger.error("DALL·E 2 Edit response did not contain expected data.")
                raise HTTPException(
                    status_code=500,
                    detail={"type": "api_error", "message": "DALL·E 2 Edit response missing data."}
                )
            
            image_url = response.data[0].url
            logger.info(f"DALL·E 2 Edit image generated successfully: {image_url}")
            
            return {
                "success": True,
                "images": [{"url": image_url, "prompt_used": payload.prompt}],
                "metadata": {
                    "mode_used": "arrange_image",
                    "settings_applied": api_params 
                }
            }

        except HTTPException as http_exc: # Re-raise HTTPExceptions directly
            raise http_exc
        except Exception as e:
            logger.error(f"DALL·E 2 Edit API call failed: {e}", exc_info=True)
            error_details = {"error_type": type(e).__name__, "message": str(e)}
            if hasattr(e, 'body'): # For OpenAI specific errors
                error_details['body'] = e.body
            
            raise HTTPException(
                status_code=500,
                detail={"type": "api_error", "message": "Failed to edit image with DALL·E 2.", "details": error_details}
            )

    elif payload.generation_mode == "generate_from_image":
        logger.info("Processing 'generate_from_image' (GPT-4V + DALL·E 3) request.")
        if not image_file: # This check is redundant due to the FastAPI File(...) default, but good for explicit check
            raise HTTPException(status_code=400, detail="Image file is required for 'generate_from_image' mode.")

        try:
            image_bytes = await image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            mime_type = image_file.content_type
            if not mime_type or mime_type == "application/octet-stream": # Fallback if content_type is generic
                filename = image_file.filename or ""
                if filename.lower().endswith(".png"):
                    mime_type = "image/png"
                elif filename.lower().endswith((".jpg", ".jpeg")):
                    mime_type = "image/jpeg"
                elif filename.lower().endswith(".webp"):
                    mime_type = "image/webp"
                else: # Default if extension is unknown or not provided
                    mime_type = "image/png" 
            
            logger.info(f"Using MIME type: {mime_type} for GPT-4V input from file: {image_file.filename}")

            gpt4v_prompt_instruction = (
                f"Analyze the style, content, and composition of the provided image. "
                f"Then, generate a detailed and specific DALL·E 3 prompt that captures the essence of this image "
                f"but reinterprets it or creates a new scene based on the following user request: '{payload.prompt}'. "
                f"The DALL·E 3 prompt should be suitable for generating a new image that harmonizes the uploaded image's feel with the user's request. "
                f"Only output the DALL·E 3 prompt itself, without any surrounding text or explanation."
            )
            
            gpt4v_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": gpt4v_prompt_instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                    ],
                }
            ]
            logger.debug("Sending request to GPT-4V (gpt-4o) for DALL·E prompt generation.")

            gpt4v_response = await openai_client.chat.completions.create(
                model="gpt-4o", # Using gpt-4o as specified
                messages=gpt4v_messages, # type: ignore
                max_tokens=350 
            )

            if not gpt4v_response.choices or not gpt4v_response.choices[0].message or not gpt4v_response.choices[0].message.content:
                logger.error("GPT-4V did not return a valid prompt.")
                raise HTTPException(status_code=500, detail={"type": "api_error", "message": "Failed to generate DALL·E prompt using GPT-4V."})
            
            dalle_prompt_from_gpt4v = gpt4v_response.choices[0].message.content.strip()
            if not dalle_prompt_from_gpt4v:
                logger.error("GPT-4V returned an empty prompt.")
                raise HTTPException(status_code=500, detail={"type": "api_error", "message": "GPT-4V generated an empty DALL·E prompt."})
            
            logger.info(f"DALL·E 3 prompt generated by GPT-4V: {dalle_prompt_from_gpt4v}")

            # DALL·E 3 Image Generation using the new prompt
            valid_sizes_dalle3 = ["1024x1024", "1024x1792", "1792x1024"]
            dalle3_size = payload.image_settings.size if payload.image_settings.size in valid_sizes_dalle3 else "1024x1024"
            valid_qualities_dalle3 = ["standard", "hd"]
            dalle3_quality = payload.image_settings.quality if payload.image_settings.quality in valid_qualities_dalle3 else "standard"
            valid_styles_dalle3 = ["vivid", "natural"]
            dalle3_style = payload.image_settings.style if payload.image_settings.style in valid_styles_dalle3 else "vivid"

            api_params_dalle3 = {
                "model": "dall-e-3",
                "prompt": dalle_prompt_from_gpt4v,
                "n": 1,
                "size": dalle3_size,
                "quality": dalle3_quality,
                "style": dalle3_style,
                "response_format": "url",
            }
            logger.debug(f"DALL·E 3 API parameters (generated from GPT-4V prompt): {api_params_dalle3}")
            
            dalle3_response = await openai_client.images.generate(**api_params_dalle3)

            if not dalle3_response.data or not dalle3_response.data[0].url:
                logger.error("DALL·E 3 (after GPT-4V) response did not contain expected data.")
                raise HTTPException(status_code=500, detail={"type": "api_error", "message": "DALL·E 3 (after GPT-4V) response missing data."})

            final_image_url = dalle3_response.data[0].url
            logger.info(f"DALL·E 3 image generated successfully from GPT-4V prompt: {final_image_url}")

            return {
                "success": True,
                "images": [{"url": final_image_url, "prompt_used": dalle_prompt_from_gpt4v}],
                "metadata": {
                    "mode_used": "generate_from_image",
                    "original_user_prompt": payload.prompt,
                    "settings_applied": api_params_dalle3
                }
            }

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Error in 'generate_from_image' mode: {e}", exc_info=True)
            error_details = {"error_type": type(e).__name__, "message": str(e)}
            if hasattr(e, 'body'):
                error_details['body'] = e.body
            raise HTTPException(
                status_code=500,
                detail={"type": "api_error", "message": "Failed in 'generate_from_image' mode.", "details": error_details}
            )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid generation_mode: {payload.generation_mode}")

    # Dummy response for now
    return {
        "success": True,
        "message": f"Request received for mode: {payload.generation_mode}",
        "images": [{"url": "placeholder_image.png", "prompt_used": payload.prompt}],
        "metadata": {
            "mode_used": payload.generation_mode,
            "settings_applied": payload.image_settings.model_dump()
        }
    }
