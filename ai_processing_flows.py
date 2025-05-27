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
            error="Perplexity client not initialized.",
        )
        return response_shell
    if not claude_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Claude (claude-opus-4-20250514)",
            error="Claude client not initialized.",
        )
        return response_shell

    from datetime import datetime
    from main import get_perplexity_response, get_claude_response

    current_dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    search_prompt = (
        f"{current_dt} 時点での最新情報を調べてください。テーマは以下の通りです：{original_prompt}\n"
        "この回答では、以下の条件を厳守してください：\n"
        "・文字数は1000文字以上\n"
        "・過去の出力と内容が重複しないこと（別視点・追加情報に集中）\n"
        "・明確で読みやすい日本語で書くこと\n"
    )

    if chat_history_for_ai:
        history_lines = []
        for msg in chat_history_for_ai:
            role = "ユーザー" if msg.get("role") == "user" else "AI"
            history_lines.append(f"{role}: {msg.get('content','')}")
        search_prompt += "\n\n[参考: これまでの会話履歴]\n" + "\n".join(history_lines)

    step1_res_perplexity = await get_perplexity_response(
        prompt_for_perplexity=search_prompt,
        model="sonar-reasoning-pro",
        user_memories=user_memories,
        initial_user_prompt=initial_user_prompt_for_session,
    )
    result_text = step1_res_perplexity.response or ""

    system_prompt = initial_user_prompt_for_session or ""

    def dedup_lines(text: str) -> str:
        seen = set()
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                lines.append(line)
                seen.add(stripped)
        return "\n".join(lines)

    result_text = dedup_lines(result_text)

    if len(result_text) < 1000 and not step1_res_perplexity.error:
        extra_res = await get_perplexity_response(
            prompt_for_perplexity=f"前回の情報と重複しない新しい視点から、さらに詳しく、1000文字以上で説明してください：\n{search_prompt}",
            model="sonar-reasoning-pro",
            user_memories=user_memories,
            initial_user_prompt=initial_user_prompt_for_session,
        )
        if extra_res.response:
            result_text += "\n" + extra_res.response
            result_text = dedup_lines(result_text)

    response_shell.step4_comprehensive_answer_perplexity = schemas.IndividualAIResponse(
        source="Perplexity (sonar-reasoning-pro)",
        response=result_text if not step1_res_perplexity.error else None,
        error=step1_res_perplexity.error,
    )

    system_prompt = (system_prompt + "\n\n" if system_prompt else "") + "特に口調を柔らかく、親しみやすい表現でまとめてください。"

    summary_prompt = (
        "以下の情報をもとに、魅力的で構成の整った日本語の文章にしてください。\n"
        "内容は削らず、むしろ必要に応じて補足しながら3000文字以上にしてください。\n"
        "読み手の興味を引く導入と、自然な結論部分も含めてください。\n\n" + result_text
    )

    step2_res_claude = await get_claude_response(
        prompt_text=summary_prompt,
        system_instruction=system_prompt,
        model="claude-opus-4-20250514",
        chat_history=chat_history_for_ai,
        initial_user_prompt=initial_user_prompt_for_session,
        user_memories=user_memories,
    )

    response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
        source="Claude (claude-opus-4-20250514)",
        response=step2_res_claude.response,
        error=step2_res_claude.error,
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

    result_text = await run_in_threadpool(
        query_perplexity, perplexity_client, original_prompt
    )
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
