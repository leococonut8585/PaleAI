# AI processing flow functions extracted from main.py

from typing import List, Dict, Optional
from fastapi import Request
import logging # Ensure logging is imported

import schemas

# Import AI helper functions from main
# Note: This can create circular dependencies if main.py also imports from this file.
# A better structure would be to have AI helpers in a separate utility module.
# For this exercise, proceeding as requested.
try:
    from main import get_perplexity_response, get_gemini_response, get_openai_response, get_cohere_response, get_claude_response
except ImportError: # pragma: no cover
    # This might happen during initial linting or if main.py is not in PYTHONPATH
    # For the purpose of this tool, we assume main.py will be available at runtime.
    # Fallback for linters or type checkers if main is not directly accessible here.
    async def get_perplexity_response(*args, **kwargs): raise NotImplementedError("get_perplexity_response not available")
    async def get_gemini_response(*args, **kwargs): raise NotImplementedError("get_gemini_response not available")
    async def get_openai_response(*args, **kwargs): raise NotImplementedError("get_openai_response not available")
    async def get_cohere_response(*args, **kwargs): raise NotImplementedError("get_cohere_response not available")
    async def get_claude_response(*args, **kwargs): raise NotImplementedError("get_claude_response not available")


async def run_quality_chat_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    """High quality mode using Perplexity then Claude."""
    logger = logging.getLogger(__name__)
    logger.info("Executing run_quality_chat_mode_flow")

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
            source="Claude (claude-opus-4-20250514)", # Corrected model name
            error="Claude client not initialized.",
        )
        return response_shell

    from datetime import datetime
    # from main import get_perplexity_response, get_claude_response # Already imported at top

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
        model="sonar-reasoning-pro", # Corrected model
        user_memories=user_memories,
        initial_user_prompt=initial_user_prompt_for_session,
    )
    result_text = step1_res_perplexity.response or ""

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
            model="sonar-reasoning-pro", # Corrected model
            user_memories=user_memories,
            initial_user_prompt=initial_user_prompt_for_session,
        )
        if extra_res.response:
            result_text += "\n" + extra_res.response
            result_text = dedup_lines(result_text)

    response_shell.step4_comprehensive_answer_perplexity = schemas.IndividualAIResponse(
        source="Perplexity (sonar-reasoning-pro)", # Corrected model
        response=result_text if not step1_res_perplexity.error else None,
        error=step1_res_perplexity.error,
    )

    system_prompt = initial_user_prompt_for_session or ""
    system_prompt = (
        system_prompt + "\n\n" if system_prompt else ""
    ) + "特に口調を柔らかく、親しみやすい表現でまとめてください。"

    summary_prompt = (
        "以下の情報をもとに、魅力的で構成の整った日本語の文章にしてください。\n"
        "内容は削らず、むしろ必要に応じて補足しながら3000文字以上にしてください。\n"
        "読み手の興味を引く導入と、自然な結論部分も含めてください。\n\n" + result_text
    )

    step2_res_claude = await get_claude_response(
        prompt_text=summary_prompt,
        system_instruction=system_prompt,
        model="claude-opus-4-20250514", # Corrected model name
        chat_history=chat_history_for_ai,
        initial_user_prompt=initial_user_prompt_for_session,
        user_memories=user_memories,
    )

    response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
        source="Claude (claude-opus-4-20250514)", # Corrected model name
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
    logger = logging.getLogger(__name__)
    logger.info("Executing run_deep_search_flow")
    
    perplexity_client = request.app.state.perplexity_sync_client
    if not perplexity_client:
        response_shell.search_summary_text = "Perplexity client not initialized."
        # Also populate the main response field for consistency
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search",
            error="Perplexity client not initialized."
        )
        response_shell.overall_error = "Perplexity client not initialized."
        return response_shell

    from fastapi.concurrency import run_in_threadpool

    def query_perplexity(client, prompt):
        try:
            client.model = "sonar-reasoning-pro" # Ensure model is set if client is shared
            return client.query(prompt)
        except Exception as exc:  # pragma: no cover - depends on external API
            logger.error(f"Perplexity query error in run_deep_search_flow: {exc}")
            return f"Perplexity error: {exc}"

    result_data = await run_in_threadpool(
        query_perplexity, perplexity_client, original_prompt
    )
    
    if isinstance(result_data, str) and result_data.startswith("Perplexity error:"):
        response_shell.search_summary_text = result_data
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search", error=result_data
        )
        response_shell.overall_error = result_data
    elif hasattr(result_data, 'answer') and result_data.answer is not None: # Assuming successful response structure
        response_text = str(result_data.answer)
        response_shell.search_summary_text = response_text
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search (sonar-reasoning-pro)", response=response_text
        )
    else: # Unexpected structure
        error_msg = f"Perplexity returned unexpected data structure: {type(result_data)}"
        logger.error(error_msg)
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search", error=error_msg
        )
        response_shell.overall_error = error_msg
        
    return response_shell


async def run_ultra_search_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    """Placeholder ultra search flow, now using get_gemini_response for consistency."""
    logger = logging.getLogger(__name__)
    logger.info("Executing run_ultra_search_flow")

    # Using the get_gemini_response helper for consistency
    gemini_res = await get_gemini_response(
        request=request,
        prompt_text=original_prompt,
        # system_instruction="You are in Ultra Search mode. Provide a comprehensive and detailed answer.", # Optional: specific system prompt
        model_name="gemini-2.5-pro-preview-05-06", # Example model
        chat_history=chat_history_for_ai,
        initial_user_prompt=initial_user_prompt_for_session,
        user_memories=user_memories
    )

    response_shell.step7_final_answer_v2_openai = gemini_res
    if gemini_res.error:
        response_shell.overall_error = f"UltraSearch (Gemini) failed: {gemini_res.error}"
        response_shell.search_summary_text = gemini_res.error
        logger.error(f"UltraSearch - FAILED: {gemini_res.error}")
    else:
        response_shell.search_summary_text = gemini_res.response
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {gemini_res.response[:100] if gemini_res.response else 'None'}...")
        
    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}\n\n【書式指示】\n各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください。重要な箇所やタイトルにはMarkdownの見出しを使用してください。"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
- 各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください。
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を適切に使用してください。
- 表を作成する場合は、Markdown形式で記述してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}\n\n【書式指示】\n要約、分類、キーワードは、それぞれMarkdownの見出しを使って区切ってください。リスト項目は改行を適切に使用し、視認性を高めてください。"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]=======
    """Placeholder ultra search flow."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    gemini_model = request.app.state.gemini_pro_model
    if not gemini_model:
        error_msg = "Gemini client not initialized."
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.search_summary_text = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        return response_shell

    try:
        # Using the direct model call as per the original structure of this placeholder
        res = await gemini_model.generate_content_async(original_prompt) 
        text = res.text if hasattr(res, "text") else str(res)
        
        logger.info(f"UltraSearch - SUCCESS. Output (truncated): {text[:100] if text else 'None'}...")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            response=text
        )
        response_shell.search_summary_text = text # Keep for compatibility

    except Exception as e:  # pragma: no cover - depends on external API
        error_msg = f"Gemini error: {e}"
        logger.error(f"UltraSearch - FAILED: {error_msg}")
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Gemini (UltraSearch Placeholder)",
            error=error_msg
        )
        response_shell.overall_error = error_msg
        response_shell.search_summary_text = error_msg

    return response_shell


# --- START OF NEW SUPERWRITING FLOWS ---

async def run_super_writing_orchestrator_flow(
    original_prompt: str,
    genre: Optional[str],
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Super Writing Orchestrator: Genre='{genre}', Prompt='{original_prompt[:100]}...'")

    if genre == "longform_composition":
        return await run_sw_longform_composition_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request
        )
    else:
        error_msg = f"Unknown genre: {genre} for Super Writing Mode."
        logger.error(error_msg) 
        response_shell.overall_error = error_msg
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Super Writing Orchestrator",
            error=error_msg,
        )
        return response_shell


async def run_sw_longform_composition_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int], 
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Longform - Start. Theme: {original_prompt[:100]}..., Desired Chars: {desired_char_count}")
    
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Longform - Step 1: Perplexity - Getting initial sources")
    prompt_step1 = f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}"
    step4_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response:
        response_shell.overall_error = "SuperWriting (Longform) - Step 4 Perplexity (recursive) failed."
        response_shell.step7_final_answer_v2_openai = step4_res
        logger.error(f"SW Longform - Step 4 FAILED: {step4_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 4 SUCCESS. Output (truncated): {step4_res.response[:100]if step4_res.response else 'None'}...")
    
    all_text_for_cohere = f"Source 1: {step1_res.response}\n\nSource 2 (additional): {step4_res.response}\n\nStructured ideas: {step2_res.response}\n\nFocus/Questions: {step3_res.response}"
    
    # Step 5: Cohere
    logger.info(f"SW Longform - Step 5: Cohere - Refinement/Deduplication")
    prompt_step5 = f"以下の情報を分析し、重複を排除しつつ、主要な論点を構造的に整理してください:\n{all_text_for_cohere}"
    step5_res = await get_cohere_response(
        prompt_text=prompt_step5, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = ""
    if step5_res.error or not step5_res.response:
        logger.warning(f"SW Longform - Step 5 Cohere (optional) FAILED or no response: {step5_res.error}. Proceeding with data before Cohere.")
        final_data_for_claude = all_text_for_cohere 
    else:
        logger.info(f"SW Longform - Step 5 SUCCESS. Output (truncated): {step5_res.response[:100]if step5_res.response else 'None'}...")
        final_data_for_claude = step5_res.response

    # Step 6: Claude
    logger.info(f"SW Longform - Step 6: Claude - Final composition")
    claude_system_prompt = f"""
あなたは「構成の設計・分割執筆・統合・整形」まで一人で行う自己完結型AI執筆者です。
以下のテーマについて、合計{desired_char_count or 30000}文字以上の日本語文書を執筆してください。

【テーマ】
{original_prompt}

【収集済みの情報と構成案】
{final_data_for_claude}

【手順】
1. まず、全体構成（章立て）を考案し、それぞれにおおよその文字数を割り振ってください（例：3000文字〜6000文字など）。
2. 次に、各章ごとに順番に1000文字以上ずつ執筆してください。章ごとに自然な区切りと一貫性を持たせてください。
3. 書き終えた各章は、そのまま順次並べてください（ユーザーの操作は必要ありません）。
4. すべての章を書き終えたら、自動的に本文全体を一つの大文書として整え直し、文体・流れ・構成を統一してください（重複・脱線・誤字脱字がないか確認してください）。
5. 文字数が合計で{desired_char_count or 30000}文字に満たない場合は、自分で章を追加したり、内容を補足してでも{desired_char_count or 30000}文字以上にしてください。

【執筆ルール】
- 日本語で書くこと
- 読みやすく、知的で魅力的な文体を使用すること
- 段落ごとの改行を行うこと (各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください)
- 章や節のタイトルにはMarkdownの見出し（例: ## タイトル）を使用してください。
- 内容の正確性と一貫性を重視すること
- 出力の途中で止まった場合は、自ら続きを再開し、最終的に{desired_char_count or 30000}文字以上の完全な一貫した文書として完了させること

【参考データ】
- 過去のチャット履歴がある場合はそれも参考にしてください (渡された履歴に含まれています)
- ユーザーメモリに絶対指示がある場合は、それを最優先で反映してください (渡されたユーザーメモリに含まれています)
- 上記が存在しない場合は、そのまま自律的に進行してください

【注意】
- トークン制限や文字制限を一切気にせず、最高の品質で書き切ってください (最大トークン数は設定済みです)
- ユーザーの操作は一切不要です。あなたがすべて自動で判断・執筆・整形してください
"""
    step6_res = await get_claude_response(
        prompt_text=claude_system_prompt, 
        system_instruction=initial_user_prompt_for_session, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error:
        response_shell.overall_error = f"SuperWriting (Longform) - Step 6 Claude failed: {step6_res.error}"
        logger.error(f"SW Longform - Step 6 FAILED: {step6_res.error}")
    else:
        logger.info(f"SW Longform - Step 6 SUCCESS. Final output generated.")
    
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_short_text_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Short Text - Start. Theme: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Short Text - Step 1: Perplexity - Getting info")
    prompt_step1 = f"テーマ「{original_prompt}」について、簡潔かつ重要な情報を調べてください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}"
    step2_res = await get_claude_response(
        prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error:
        response_shell.overall_error = f"SuperWriting (Short Text) - Claude failed: {step2_res.error}"
        logger.error(f"SW Short Text - Step 2 FAILED: {step2_res.error}")
    else:
        logger.info(f"SW Short Text - Step 2 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_thesis_report_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Thesis/Report - Start. Topic: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Thesis/Report - Step 1: Perplexity - Initial research")
    prompt_step1 = f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。"
    step1_res = await get_perplexity_response(
        prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    step3_res = await get_openai_response(
        prompt_text=prompt_step3, model="gpt-4o", user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response:
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Thesis/Report - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100] if step3_res.response else 'None'}...")

    # Step 4: Claude
    logger.info(f"SW Thesis/Report - Step 4: Claude - Drafting report")
    prompt_step4 = f"""
以下のテーマ、調査結果、構成案、および深掘りされた論点に基づき、学術的な論文・レポート本文を執筆してください。
テーマ: {original_prompt}

初期調査結果の要約:
{step1_res.response[:2000] if step1_res.response else ''}... 

作成された構成案:
{step2_res.response}

各論点の深掘りと追加の問い:
{step3_res.response}

執筆指示:
- 上記情報を統合し、論理的で一貫性のある論文・レポートを作成してください。
- 適切な導入、本論（各章・節）、結論を含めてください。
- 必要に応じて情報を補足し、議論を深めてください。
- 学術的な文体を使用し、明確かつ客観的に記述してください。
- 段落ごとの改行、Markdownの見出しを適切に使用してください。
"""
    step4_res = await get_claude_response(
        prompt_text=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error:
        response_shell.overall_error = f"SuperWriting (Thesis/Report) - Step 4 Claude failed: {step4_res.error}"
        logger.error(f"SW Thesis/Report - Step 4 FAILED: {step4_res.error}")
    else:
        logger.info(f"SW Thesis/Report - Step 4 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell


async def run_sw_summary_classification_flow(
    original_prompt: str, 
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"SW Summary/Classify - Start. Input: {original_prompt[:100]}...")
    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Step 1: Perplexity
    logger.info(f"SW Summary/Classify - Step 1: Perplexity (optional)")
    is_text_input = len(original_prompt) > 500 
    
    perplexity_output_text = original_prompt
    step1_res: Optional[schemas.IndividualAIResponse] = None 
    if not is_text_input:
        prompt_step1_perplexity = f"テーマ「{original_prompt}」に関する情報を収集してください。"
        step1_res = await get_perplexity_response(
            prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response:
            response_shell.overall_error = "SuperWriting (Summary/Classify) - Perplexity failed."
            response_shell.step7_final_answer_v2_openai = step1_res
            logger.error(f"SW Summary/Classify - Step 1 FAILED: {step1_res.error}")
            response_shell.ultra_writing_mode_details = intermediate_steps_details
            return response_shell
        perplexity_output_text = step1_res.response
        logger.info(f"SW Summary/Classify - Step 1 SUCCESS (topic research). Output (truncated): {perplexity_output_text[:100] if perplexity_output_text else 'None'}...")
    else:
        logger.info(f"SW Summary/Classify - Step 1 SKIPPED (input assumed to be text).")
    
    # Step 2: Gemini
    logger.info(f"SW Summary/Classify - Step 2: Gemini - Structuring input")
    prompt_step2_gemini = f"以下の情報を分析し、主要なトピックやセクションに構造化してください。もしこれが単一のテキストであれば、その要点を整理してください:\n{perplexity_output_text}"
    step2_res = await get_gemini_response(
        request=request, prompt_text=prompt_step2_gemini, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response:
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}"
    step3_res = await get_cohere_response(
        prompt_text=prompt_step3_cohere, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session, chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error:
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

[end of ai_processing_flows.py]
