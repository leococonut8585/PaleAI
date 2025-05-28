import base64
import logging 
import shutil
from pathlib import Path 
import httpx
import asyncio
from typing import List, Optional, Any, Dict 
from fastapi import Request

from utils.openai_client import openai_client

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
    logger.debug("実際に使ったプロンプト: %s", prompt)
    params = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": "1024x1024",
        "n": 1,
        "quality": "standard",
        "style": "vivid",
    }
    logger.debug("DALL·E request params: %s", params)
    try:
        resp = await openai_client.images.generate(**params)
        logger.debug("DALL·E raw response: %s", resp)
        url = resp.data[0].url
        logger.info("DALL·E image URL: %s", url)
        async with httpx.AsyncClient() as cx:
            content = (await cx.get(url)).content
        return content
    except Exception as e:
        logger.error("DALL·E API呼び出しでエラー発生: %s", e)
        if hasattr(e, "response") and hasattr(e.response, "text"):
            logger.error("DALL·E APIエラーレスポンス (Text): %s", e.response.text)
            try:
                error_detail_json = e.response.json()
                logger.error(
                    "DALL·E APIエラー詳細 (JSON from response): %s",
                    error_detail_json,
                )
            except Exception:
                logger.error(
                    "DALL·E APIエラーレスポンスのJSONデコードに失敗 (from response.text)"
                )
        if hasattr(e, "body") and e.body:
            logger.error("DALL·E APIエラー詳細 (e.body): %s", e.body)
        if hasattr(e, "code"):
            logger.error(f"OpenAI Specific Error Code: {e.code}")
        if hasattr(e, "message"):
            logger.error(f"OpenAI Specific Error Message: {e.message}")
        if hasattr(e, "type"):
            logger.error(f"OpenAI Specific Error Type: {e.type}")
        raise

async def generate_and_save(
    c1_hex: str,
    c2_hex: str,
    gender: str,
    user_id: int,
    openai_api_client: Any, # Changed from request: Request
    file_name: Optional[str] = None,
) -> None:
    """Generate a profile image using a GPT-4V styled DALL·E 3 prompt and save it.

    Parameters
    ----------
    c1_hex, c2_hex, gender : str
        Color values and gender used to build the prompt.
    user_id : int
        User identifier.
    openai_api_client : Any
        An instance of the OpenAI API client.
    file_name : str | None
        Optional custom file name (without directory) for the generated image.
        When omitted, ``"{user_id}.png"`` is used.
    """

    c1_name = _hex_to_color_name(c1_hex)
    c2_name = _hex_to_color_name(c2_hex)
    
    reference_image_filenames = ["Answer.png", "Default.png", "Thinking.png"]
    img_bytes: Optional[bytes] = None

    try:
        logger.info(f"Attempting new GPT-4V based prompt generation for user {user_id}.")
        dalle_prompt_from_gpt4v = await _generate_styled_dalle_prompt_with_gpt4v(
            user_color1_name=c1_name,
            user_color2_name=c2_name,
            user_gender=gender,
            reference_filenames=reference_image_filenames,
            openai_api_client=openai_api_client # Pass the client instance
        )

        if dalle_prompt_from_gpt4v:
            logger.info(f"Successfully generated DALL-E prompt via GPT-4V for user {user_id}.")
            img_bytes = await _gen_with_dalle(dalle_prompt_from_gpt4v) 
        else:
            logger.warning(f"Failed to generate DALL-E prompt via GPT-4V for user {user_id}. Proceeding to fallback.")
    except Exception as e_gpt4v_dalle:
        logger.error(f"Error during GPT-4V prompt generation or DALL-E call for user {user_id}: {e_gpt4v_dalle}")
        # img_bytes will remain None, leading to fallback

    if img_bytes:
        Path("static/profile").mkdir(exist_ok=True, parents=True)
        out_name = file_name or f"{user_id}.png"
        (Path("static/profile") / out_name).write_bytes(img_bytes)
        logger.info(f"Saved new AI-generated image for user {user_id} to {out_name} using GPT-4V styled prompt.")
    else:
        logger.error(f"All image generation attempts failed for user {user_id} (including new GPT-4V path). Using default image.")
        Path("static/profile").mkdir(exist_ok=True, parents=True)
        out_name = file_name or f"{user_id}.png"
        try:
            shutil.copy("static/pic/Default.png", str(Path("static/profile") / out_name))
            logger.info(f"Copied default image for user {user_id} to {out_name}.")
        except Exception as e_copy:
            logger.error(f"Failed to copy default image for user {user_id}: {e_copy}")
            # If even copying default fails, there's not much more to do here for image generation.


async def _generate_styled_dalle_prompt_with_gpt4v(
    user_color1_name: str, 
    user_color2_name: str, 
    user_gender: str, 
    reference_filenames: List[str], 
    openai_api_client: Any # Changed from request: Request
) -> Optional[str]:
    logger = logging.getLogger(__name__)
    
    # openai_api_client is now passed directly as a parameter
    if openai_api_client is None:
        logger.error("OpenAI client instance is None.") # Updated error message slightly
        return None

    image_content_parts: List[Dict[str, Any]] = []
    for filename in reference_filenames:
        image_path = Path("static/pic/") / filename # Changed directory here
        try:
            image_bytes = image_path.read_bytes()
            base64_image_string = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
            image_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image_string}"}})
        except FileNotFoundError:
            logger.warning(f"Reference image {image_path} not found, skipping.")
        except Exception as e_file:
            logger.error(f"Error processing reference image {image_path}: {e_file}")

    if not image_content_parts:
        logger.error("No valid reference images were loaded or processed.")
        return None

    gender_specific_details = ""
    if user_gender == "女性":
        gender_specific_details = "女性的な顔つき、細い体、長いまつげ、魅力的なポーズをしています。"
    elif user_gender == "男性":
        gender_specific_details = "男性的な顔つき、太い体、短い太いまゆげ、力強いポーズをしています。"
    else:
        gender_specific_details = "指定なし。中性的な特徴で。"

    user_message_content_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "以下の参照画像の芸術的スタイル（線画、配色、陰影技術、キャラクターデザインのモチーフ、全体的な雰囲気、芸術的「タッチ」など）を詳細に分析してください。\n\n参照画像群:"}
    ]
    user_message_content_parts.extend(image_content_parts) 
    
    final_text_instruction = (
        "\n\n分析したスタイルに基づき、以下の要素を組み込んだ「サイボーグモンキーのマスコットキャラクターのプロフィールアイコン、透明背景」を描画するための、詳細で具体的なDALL·E 3プロンプトを生成してください："
        f"\n- 主要な体色として「{user_color1_name}」と「{user_color2_name}」を使用。"
        f"\n- 性別特徴: {gender_specific_details}"
        "\n- スタイル特徴: シンプルな線画、太い黒のアウトライン、フラットで明るい色彩、最小限の陰影。"
        "\n- サイボーグ特徴: 腕の関節、首、ヘッドバンドなど、見える機械部品。"
        "\n- 表情その他: 口を開けた陽気な笑顔、フレンドリーで親しみやすく遊び心のある外見。"
        "\n- 禁止事項: 文字、ウォーターマーク、余分な図形やバー、背景は一切含めない。"
        "\n\n生成するDALL·E 3プロンプトは、参照画像のスタイルを忠実に再現しつつ、これらの新しい要素を盛り込んだものにしてください。"
        "単に参照画像を描写するのではなく、そのスタイルで新しいキャラクターを描くためのプロンプトを作成してください。"
        "DALL·E 3プロンプトのみを返答し、それ以外の前置きや解説は不要です。"
    )
    user_message_content_parts.append({"type": "text", "text": final_text_instruction})

    messages_for_gpt4v: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "あなたは高度なアートスタイルアナリストであり、創造的なプロンプトエンジニアです。提供された参照画像を分析し、そのスタイルを反映しつつ、指定された新しい内容の画像を生成するための詳細なDALL·E 3プロンプトを作成する任務を負っています。"
        },
        {
            "role": "user",
            "content": user_message_content_parts
        }
    ]
    
    try:
        logger.info(f"Calling GPT-4V (gpt-4o) to generate DALL-E prompt. User colors: {user_color1_name}, {user_color2_name}. Gender: {user_gender}.")
        gpt4v_response = await openai_api_client.chat.completions.create(
            model="gpt-4o", 
            messages=messages_for_gpt4v, # type: ignore[arg-type] 
            max_tokens=700, 
            temperature=0.5 
        )
        if gpt4v_response.choices and gpt4v_response.choices[0].message and gpt4v_response.choices[0].message.content:
            dalle_prompt = gpt4v_response.choices[0].message.content.strip()
            if dalle_prompt: 
                 logger.info(f"GPT-4V generated DALL-E prompt (first 100 chars): {dalle_prompt[:100]}...")
                 return dalle_prompt
            else:
                 logger.error("GPT-4V generated a blank DALL-E prompt after stripping.")
                 return None
        else:
            logger.error("GPT-4V returned an unexpected response structure or empty content for DALL-E prompt generation. Response: %s", str(gpt4v_response))
            return None
    except Exception as e:
        logger.error(f"Error calling GPT-4V for DALL-E prompt generation: {e}")
        if hasattr(e, "response") and e.response is not None and hasattr(e.response, "text"): 
             logger.error(f"GPT-4V API Error details: {e.response.text}") # type: ignore[attr-defined]
        elif hasattr(e, "body") and e.body is not None: 
             logger.error(f"GPT-4V API Error body: {e.body}")
        return None
