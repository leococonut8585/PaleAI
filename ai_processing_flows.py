# AI processing flow functions extracted from main.py
# These are simplified placeholders to avoid circular imports

from typing import List, Dict, Optional
from fastapi import Request

import schemas


async def run_quality_chat_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    """High quality mode using Perplexity then Claude."""

    perplexity_client = request.app.state.perplexity_sync_client
    claude_client = request.app.state.anthropic_client

    if not perplexity_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity (sonar-reasoning-pro)",
            error="Perplexity client not initialized."
        )
        return response_shell
    if not claude_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Claude (claude-opus-4-20250514)", error="Claude client not initialized."
        )
        return response_shell

    from datetime import datetime
    from fastapi.concurrency import run_in_threadpool

    current_dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    search_prompt = f"{current_dt} 時点の最新情報を調べてください。{original_prompt}"

    def query_perplexity(client, prompt):
        try:
            client.model = "sonar-reasoning-pro"
            return client.query(prompt)
        except Exception as exc:
            return f"Perplexity error: {exc}"

    result_text = await run_in_threadpool(query_perplexity, perplexity_client, search_prompt)
    if not isinstance(result_text, str):
        result_text = str(result_text)

    if len(result_text) < 3000:
        extra = await run_in_threadpool(
            query_perplexity,
            perplexity_client,
            f"さらに詳しく、3000文字以上で: {search_prompt}",
        )
        if isinstance(extra, str):
            result_text += "\n" + extra

    response_shell.step4_comprehensive_answer_perplexity = schemas.IndividualAIResponse(
        source="Perplexity (sonar-reasoning-pro)", response=result_text
    )

    messages = []
    if initial_user_prompt_for_session:
        messages.append({"role": "system", "content": initial_user_prompt_for_session})

    summary_prompt = (
        "以下の情報を魅力的で読みやすい文章に整形してください。内容を削らず日本語でまとめてください。\n\n"
        + result_text
    )
    messages.append({"role": "user", "content": summary_prompt})

    try:
        res = await claude_client.messages.create(
            model="claude-opus-4-20250514",
            messages=messages,
            temperature=0.6,
        )
        text = ""
        if res and hasattr(res, "content") and isinstance(res.content, list):
            for block in res.content:
                if hasattr(block, "text"):
                    text += block.text
        elif hasattr(res, "text"):
            text = res.text
        else:
            text = str(res)

        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Claude (claude-opus-4-20250514)", response=text
        )
    except Exception as e:  # pragma: no cover - network errors not testable
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Claude (claude-opus-4-20250514)", error=str(e)
        )

    return response_shell


async def run_deep_search_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    """Simplified deep search using Perplexity."""
    perplexity_client = request.app.state.perplexity_sync_client
    if not perplexity_client:
        response_shell.search_summary_text = "Perplexity client not initialized."
        return response_shell

    from fastapi.concurrency import run_in_threadpool

    def query_perplexity(client, prompt):
        try:
            client.model = "sonar-reasoning-pro"
            return client.query(prompt)
        except Exception as exc:  # pragma: no cover - depends on external API
            return f"Perplexity error: {exc}"

    result_text = await run_in_threadpool(query_perplexity, perplexity_client, original_prompt)
    response_shell.search_summary_text = str(result_text)
    return response_shell


async def run_ultra_search_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    """Placeholder ultra search flow."""
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        response_shell.search_summary_text = "Gemini client not initialized."
        return response_shell

    try:
        res = await gemini_model.generate_content(original_prompt)
        text = res.text if hasattr(res, "text") else str(res)
    except Exception as e:  # pragma: no cover - depends on external API
        text = f"Gemini error: {e}"

    response_shell.search_summary_text = text
    return response_shell
