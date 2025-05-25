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
    """Simplified quality chat flow using OpenAI."""
    openai_client = request.app.state.openai_client
    if not openai_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="OpenAI (gpt-4o)", error="OpenAI client not initialized."
        )
        return response_shell

    messages = []
    if initial_user_prompt_for_session:
        messages.append({"role": "system", "content": initial_user_prompt_for_session})
    for msg in chat_history_for_ai:
        role = msg.get("role")
        if role == "ai":
            role = "assistant"
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(msg.get("content", ""))})
    messages.append({"role": "user", "content": original_prompt})

    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.7
        )
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="OpenAI (gpt-4o)", response=res.choices[0].message.content
        )
    except Exception as e:  # pragma: no cover - network errors not testable
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="OpenAI (gpt-4o)", error=str(e)
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
            client.model = "sonar-pro"
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
