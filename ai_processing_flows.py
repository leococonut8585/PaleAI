# AI processing flow functions extracted from main.py

from typing import List, Dict, Optional
from fastapi import Request
import logging # Ensure logging is imported

import schemas

# Import AI helper functions from ai_clients
from ai_clients import (
    get_perplexity_response,
    get_gemini_response,
    get_openai_response,
    get_cohere_response,
    get_claude_response
)
# Fallback for linters or type checkers if ai_clients is not directly accessible (though it should be)
try:
    pass # Imports are now at the top level
except ImportError: # pragma: no cover
    # Fallback for linters or type checkers if main is not directly accessible here.
    # These fallbacks are no longer strictly necessary if imports are correct
    # but kept for extreme safety during tool execution, will be cleaned if successful.
    async def get_perplexity_response(*args, **kwargs): raise NotImplementedError("get_perplexity_response not available") # pragma: no cover
    async def get_gemini_response(*args, **kwargs): raise NotImplementedError("get_gemini_response not available") # pragma: no cover
    async def get_openai_response(*args, **kwargs): raise NotImplementedError("get_openai_response not available") # pragma: no cover
    async def get_cohere_response(*args, **kwargs): raise NotImplementedError("get_cohere_response not available") # pragma: no cover
    async def get_claude_response(*args, **kwargs): raise NotImplementedError("get_claude_response not available") # pragma: no cover


async def run_quality_chat_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request, # Ensure request is accepted
) -> schemas.CollaborativeResponseV2:
    """High quality mode using Perplexity then Claude."""
    logger = logging.getLogger(__name__)
    logger.info("Executing run_quality_chat_mode_flow")

    # Client states are already on request.app.state
    if not request.app.state.perplexity_sync_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity (sonar-reasoning-pro)", error="Perplexity client not initialized.",
        )
        response_shell.overall_error = "Perplexity client not initialized."
        return response_shell
    if not request.app.state.anthropic_client:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Claude (claude-opus-4-20250514)", error="Claude client not initialized.",
        )
        response_shell.overall_error = "Claude client not initialized."
        return response_shell

    from datetime import datetime
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
        request=request, # Pass request
        prompt_for_perplexity=search_prompt,
        model="sonar-reasoning-pro",
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
            request=request, # Pass request
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

    system_instruction_for_claude = "特に口調を柔らかく、親しみやすい表現でまとめてください。"
    summary_prompt = (
        "以下の情報をもとに、魅力的で構成の整った日本語の文章にしてください。\n"
        "内容は削らず、むしろ必要に応じて補足しながら3000文字以上にしてください。\n"
        "読み手の興味を引く導入と、自然な結論部分も含めてください。\n\n" + result_text
    )

    step2_res_claude = await get_claude_response(
        request=request, # Pass request
        prompt_text=summary_prompt,
        system_instruction=system_instruction_for_claude,
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
    request: Request, # Ensure request is accepted
) -> schemas.CollaborativeResponseV2:
    """Simplified deep search using Perplexity."""
    logger = logging.getLogger(__name__)
    logger.info("Executing run_deep_search_flow")
    
    # This function should call get_perplexity_response, not use client directly
    perplexity_res = await get_perplexity_response(
        request=request, # Pass request
        prompt_for_perplexity=original_prompt,
        model="sonar-reasoning-pro", # Example, or choose dynamically
        user_memories=user_memories,
        initial_user_prompt=chat_history_for_ai[0].get("content") if chat_history_for_ai and chat_history_for_ai[0].get("role") == "user" else original_prompt, # Heuristic for initial prompt
    )

    if perplexity_res.error:
        response_shell.search_summary_text = perplexity_res.error
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search", error=perplexity_res.error
        )
        response_shell.overall_error = perplexity_res.error
    else:
        response_shell.search_summary_text = perplexity_res.response
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Perplexity Deep Search (sonar-reasoning-pro)", response=perplexity_res.response, links=perplexity_res.links
        )
        
    return response_shell


async def run_ultra_search_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    request: Request, # Ensure request is accepted
) -> schemas.CollaborativeResponseV2:
    """Placeholder ultra search flow, now using get_gemini_response for consistency."""
    logger = logging.getLogger(__name__)
    logger.info("Executing run_ultra_search_flow")

    # Ensure initial_user_prompt_for_session is correctly determined or passed if needed
    # For simplicity, if not directly available, might derive from chat history or original_prompt.
    # This was missing `initial_user_prompt_for_session` in the original call from main.py for this flow if it was intended.
    # Assuming `initial_user_prompt_for_session` should be passed to this flow if available.
    # If `initial_user_prompt_for_session` is not available, we might use the first user message or original_prompt.
    derived_initial_user_prompt = chat_history_for_ai[0].get("content") if chat_history_for_ai and chat_history_for_ai[0].get("role") == "user" else original_prompt

    gemini_res = await get_gemini_response(
        request=request, # Pass request
        prompt_text=original_prompt,
        model_name="gemini-2.5-pro-preview-05-06",
        chat_history=chat_history_for_ai,
        initial_user_prompt=derived_initial_user_prompt, # Use derived or passed initial prompt
        user_memories=user_memories
    )

    response_shell.step7_final_answer_v2_openai = gemini_res # This already contains source and error/response
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
    # All sub-flows will receive the `request` object from this orchestrator.
    if genre == "longform_composition":
        # Now, longform_composition uses the new iterative flow
        return await run_iterative_super_drafting_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, desired_char_count, request
        )
    elif genre == "short_text":
        return await run_sw_short_text_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request # Pass request
        )
    elif genre == "thesis_report":
        return await run_sw_thesis_report_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request # Pass request
        )
    elif genre == "summary_classification":
        return await run_sw_summary_classification_flow(
            original_prompt, response_shell, chat_history_for_ai,
            initial_user_prompt_for_session, user_memories, request # Pass request
        )
    # Removed elif for "iterative_super_drafting" as "longform_composition" now uses it.
    else: # pragma: no cover
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
    prompt_step1 = (
        f"最新情報を3～5ソース取得してください（引用・リンク付き）。テーマ: {original_prompt}\n"
        "このステップの応答では、ウキヨザルのキャラクター性は一切含めず、客観的かつ分析的なトーンで記述してください。"
    )
    step1_res = await get_perplexity_response( # Pass request
        request=request, prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Longform) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Longform - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")
    
    # Step 2: Gemini
    logger.info(f"SW Longform - Step 2: Gemini - Structuring Perplexity output")
    prompt_step2 = f"以下の情報群から構造化された要約（章立て候補や情報群）を作成してください:\n{step1_res.response}"
    system_instruction_step2 = "このステップの応答では、ウキヨザルのキャラクター性は一切含めず、客観的かつ分析的なトーンで記述してください。"
    step2_res = await get_gemini_response( # Pass request
        request=request, 
        prompt_text=prompt_step2, 
        system_instruction=system_instruction_step2,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Longform) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Longform - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100]if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Longform - Step 3: GPT-4o - Generating questions/prompts")
    prompt_step3 = f"以下の構造情報と元のテーマ「{original_prompt}」に基づいて、「問い」（深掘りの観点提案）、非重複の検索プロンプト生成、コンテンツの焦点提案を行ってください:\n{step2_res.response}"
    system_role_description_step3 = "このステップの応答では、ウキヨザルのキャラクター性は一切含めず、客観的かつ分析的なトーンで記述してください。"
    step3_res = await get_openai_response( # Pass request
        request=request,
        prompt_text=prompt_step3, 
        model="gpt-4o", 
        system_role_description=system_role_description_step3,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Longform) - Step 3 GPT-4o failed."
        response_shell.step7_final_answer_v2_openai = step3_res
        logger.error(f"SW Longform - Step 3 FAILED: {step3_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Longform - Step 3 SUCCESS. Output (truncated): {step3_res.response[:100]if step3_res.response else 'None'}...")

    # Step 4: Perplexity (Recursive)
    logger.info(f"SW Longform - Step 4: Perplexity - Recursive search")
    prompt_step4 = (
        f"以下の「問い」や検索プロンプトに基づいて再度情報を収集してください (政治・技術・文化などの各視点ごとに追加情報収集):\n{step3_res.response}\n"
        "このステップの応答では、ウキヨザルのキャラクター性は一切含めず、客観的かつ分析的なトーンで記述してください。"
    )
    step4_res = await get_perplexity_response( # Pass request
        request=request, prompt_for_perplexity=prompt_step4, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step4_res)
    if step4_res.error or not step4_res.response: # pragma: no cover
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
    preamble_step5 = "このステップの応答では、ウキヨザルのキャラクター性は一切含めず、客観的かつ分析的なトーンで記述してください。"
    step5_res = await get_cohere_response( # Pass request
        request=request,
        prompt_text=prompt_step5, 
        preamble=preamble_step5,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step5_res)
    final_data_for_claude = "" # Initialize
    if step5_res.error or not step5_res.response: # pragma: no cover
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
    system_instruction_for_claude_step6 = (
        (initial_user_prompt_for_session + "\n\n" if initial_user_prompt_for_session else "") +
        "最終的な文章のトーンは、ユーザーの指示と執筆テーマに最も適した、知的で魅力的な文体としてください。ウキヨザルのキャラクター性は反映しないでください。"
    )
    step6_res = await get_claude_response( # Pass request
        request=request,
        prompt_text=claude_system_prompt, 
        system_instruction=system_instruction_for_claude_step6, 
        user_memories=user_memories,
        chat_history=chat_history_for_ai 
    )
    intermediate_steps_details.append(step6_res)
    response_shell.step7_final_answer_v2_openai = step6_res
    if step6_res.error: # pragma: no cover
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
    step1_res = await get_perplexity_response( # Pass request
        request=request, prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Short Text) - Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Short Text - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Short Text - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Claude
    logger.info(f"SW Short Text - Step 2: Claude - Generating short text")
    prompt_step2 = f"以下の情報をもとに、テーマ「{original_prompt}」について自然で簡潔な短文を作成してください:\n{step1_res.response}\n\n【書式指示】\n各段落やリスト項目の間には十分な改行を入れ、視認性を高めてください。重要な箇所やタイトルにはMarkdownの見出しを使用してください。"
    system_instruction_for_claude_step2 = (
        "あなたは、与えられた情報を元に、自然で簡潔な高品質の短文を作成する専門家です。"
        "文体は、ユーザーの指示とテーマに応じて適切に調整し、特に指定がない場合は中立的かつプロフェッショナルなトーンを使用してください。"
        "ウキヨザルのキャラクター性は反映しないでください。"
    )
    step2_res = await get_claude_response( # Pass request
        request=request,
        prompt_text=prompt_step2, 
        system_instruction=system_instruction_for_claude_step2,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    response_shell.step7_final_answer_v2_openai = step2_res
    if step2_res.error: # pragma: no cover
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
    prompt_step1 = (
        f"論文・レポートのテーマ「{original_prompt}」に関する最新かつ信頼性の高い情報を包括的に調査してください。\n"
        "この調査は学術的な目的のためのものです。応答は客観的かつ分析的なトーンで、ウキヨザルのキャラクター性は一切含めないでください。"
    )
    step1_res = await get_perplexity_response( # Pass request
        request=request, prompt_for_perplexity=prompt_step1, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
    )
    intermediate_steps_details.append(step1_res)
    if step1_res.error or not step1_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 1 Perplexity failed."
        response_shell.step7_final_answer_v2_openai = step1_res
        logger.error(f"SW Thesis/Report - Step 1 FAILED: {step1_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 1 SUCCESS. Output (truncated): {step1_res.response[:100] if step1_res.response else 'None'}...")

    # Step 2: Gemini
    logger.info(f"SW Thesis/Report - Step 2: Gemini - Structuring research")
    prompt_step2 = f"以下の調査結果に基づき、テーマ「{original_prompt}」の論文・レポートのための詳細な構成案（章立て、各セクションの主要な論点）を作成してください:\n{step1_res.response}"
    system_instruction_step2 = (
        "あなたは学術論文の構成案を作成する専門家です。客観的かつ構造的な提案を、プロフェッショナルなトーンで行ってください。"
        "ウキヨザルのキャラクター性は反映しないでください。"
    )
    step2_res = await get_gemini_response( # Pass request
        request=request, 
        prompt_text=prompt_step2, 
        system_instruction=system_instruction_step2,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Thesis/Report) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Thesis/Report - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Thesis/Report - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: GPT-4o
    logger.info(f"SW Thesis/Report - Step 3: GPT-4o - Refining outline and questions")
    prompt_step3 = f"テーマ「{original_prompt}」に関する以下の調査結果と構成案をレビューし、各セクションで展開すべき議論や必要な追加調査項目、論点を深めるための問いを生成してください:\n調査結果概要:\n{step1_res.response[:1000] if step1_res.response else ''}...\n\n構成案:\n{step2_res.response}"
    system_role_description_step3 = (
        "あなたは学術的な議論を深めるための問いを生成するリサーチアシスタントです。"
        "応答は分析的かつ客観的な視点から、専門的なトーンで行ってください。ウキヨザルのキャラクター性は一切含めないでください。"
    )
    step3_res = await get_openai_response( # Pass request
        request=request,
        prompt_text=prompt_step3, 
        model="gpt-4o", 
        system_role_description=system_role_description_step3,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    if step3_res.error or not step3_res.response: # pragma: no cover
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
    system_instruction_for_claude_step4 = (
        (initial_user_prompt_for_session + "\n\n" if initial_user_prompt_for_session else "") +
        "最終的な論文・レポートは、指示された通り厳密に学術的な文体で記述し、ウキヨザルのキャラクター性は一切反映しないでください。"
        "客観性と論理性を最優先してください。"
    )
    step4_res = await get_claude_response( # Pass request
        request=request,
        prompt_text=prompt_step4, 
        system_instruction=system_instruction_for_claude_step4,
        user_memories=user_memories, 
        chat_history=chat_history_for_ai
        # initial_user_prompt is part of system_instruction
    )
    intermediate_steps_details.append(step4_res)
    response_shell.step7_final_answer_v2_openai = step4_res
    if step4_res.error: # pragma: no cover
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
    is_text_input = len(original_prompt) > 500 # Threshold to decide if input is a topic or text
    
    perplexity_output_text = original_prompt # Default if input is already text
    step1_res: Optional[schemas.IndividualAIResponse] = None # Initialize
    if not is_text_input: # If input is likely a topic, do Perplexity search
        prompt_step1_perplexity = (
            f"テーマ「{original_prompt}」に関する情報を収集してください。\n"
            "この情報収集は分析目的です。応答は客観的かつ事実に即したトーンで、ウキヨザルのキャラクター性は一切含めないでください。"
        )
        step1_res = await get_perplexity_response( # Pass request
            request=request, prompt_for_perplexity=prompt_step1_perplexity, user_memories=user_memories, initial_user_prompt=initial_user_prompt_for_session
        )
        intermediate_steps_details.append(step1_res)
        if step1_res.error or not step1_res.response: # pragma: no cover
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
    system_instruction_step2 = (
        "あなたはテキストを分析し、構造化する専門家です。応答は客観的かつ論理的なトーンで、主要なトピックや要点を整理してください。"
        "ウキヨザルのキャラクター性は反映しないでください。"
    )
    step2_res = await get_gemini_response( # Pass request
        request=request, 
        prompt_text=prompt_step2_gemini, 
        system_instruction=system_instruction_step2,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step2_res)
    if step2_res.error or not step2_res.response: # pragma: no cover
        response_shell.overall_error = "SuperWriting (Summary/Classify) - Step 2 Gemini failed."
        response_shell.step7_final_answer_v2_openai = step2_res
        logger.error(f"SW Summary/Classify - Step 2 FAILED: {step2_res.error}")
        response_shell.ultra_writing_mode_details = intermediate_steps_details
        return response_shell
    logger.info(f"SW Summary/Classify - Step 2 SUCCESS. Output (truncated): {step2_res.response[:100] if step2_res.response else 'None'}...")

    # Step 3: Cohere
    logger.info(f"SW Summary/Classify - Step 3: Cohere - Summarize/Classify/Tag")
    prompt_step3_cohere = f"以下の構造化されたテキストまたは要点に基づいて、詳細な要約を作成し、主要な分類を行い、関連するキーワードやタグを抽出してください:\n{step2_res.response}\n\n【書式指示】\n要約、分類、キーワードは、それぞれMarkdownの見出しを使って区切ってください。リスト項目は改行を適切に使用し、視認性を高めてください。"
    preamble_step3 = (
        "あなたはテキストの要約、分類、キーワード抽出を行う分析AIです。"
        "応答は客観的かつ簡潔なトーンで、指定されたフォーマットに従ってください。ウキヨザルのキャラクター性は一切含めないでください。"
    )
    step3_res = await get_cohere_response( # Pass request
        request=request,
        prompt_text=prompt_step3_cohere, 
        preamble=preamble_step3,
        user_memories=user_memories, 
        initial_user_prompt=initial_user_prompt_for_session, 
        chat_history=chat_history_for_ai
    )
    intermediate_steps_details.append(step3_res)
    response_shell.step7_final_answer_v2_openai = step3_res
    if step3_res.error: # pragma: no cover
        response_shell.overall_error = f"SuperWriting (Summary/Classify) - Step 3 Cohere failed: {step3_res.error}"
        logger.error(f"SW Summary/Classify - Step 3 FAILED: {step3_res.error}")
    else:
        logger.info(f"SW Summary/Classify - Step 3 SUCCESS. Final output generated.")
        
    response_shell.ultra_writing_mode_details = intermediate_steps_details
    return response_shell

# --- END OF NEW SUPERWRITING FLOWS ---

import math

async def run_iterative_super_drafting_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    user_memories: Optional[List[schemas.UserMemoryResponse]],
    desired_char_count: Optional[int],
    request: Request,
) -> schemas.CollaborativeResponseV2:
    logger = logging.getLogger(__name__)
    logger.info(f"Executing run_iterative_super_drafting_flow for prompt: {original_prompt[:100]}...")

    intermediate_steps_details: List[schemas.IndividualAIResponse] = []

    # Initialization
    DEFAULT_TOTAL_CHARS = 100000
    MAX_CHARS_PER_CHUNK = 10000
    MIN_CHARS_PER_CHUNK = 1000

    if desired_char_count is None or desired_char_count < MIN_CHARS_PER_CHUNK:
        actual_desired_char_count = DEFAULT_TOTAL_CHARS
        logger.warning(f"Desired char count {desired_char_count} is too small or None. Using default: {actual_desired_char_count}")
    else:
        actual_desired_char_count = desired_char_count

    if actual_desired_char_count <= MAX_CHARS_PER_CHUNK:
        num_chunks = 1
        target_chars_per_chunk = actual_desired_char_count
    else:
        num_chunks = math.ceil(actual_desired_char_count / MAX_CHARS_PER_CHUNK)
        target_chars_per_chunk = math.ceil(actual_desired_char_count / num_chunks)

    if target_chars_per_chunk < MIN_CHARS_PER_CHUNK:
        target_chars_per_chunk = MIN_CHARS_PER_CHUNK
        logger.warning(f"Calculated target_chars_per_chunk {target_chars_per_chunk} was too small. Adjusted to {MIN_CHARS_PER_CHUNK}")
        # Recalculate num_chunks if target_chars_per_chunk was adjusted, to ensure total is still met
        num_chunks = math.ceil(actual_desired_char_count / target_chars_per_chunk)


    logger.info(f"Total Chars: {actual_desired_char_count}, Chunks: {num_chunks}, Target per Chunk: {target_chars_per_chunk}")

    all_generated_text = ""
    completed_chunks_data: List[schemas.IndividualAIResponse] = []

    # Iterative Generation Loop
    for current_chunk_num in range(1, num_chunks + 1):
        logger.info(f"Starting chunk {current_chunk_num}/{num_chunks}")
        current_chunk_text = ""
        max_retries_char_count = 3
        max_retries_consistency = 2

        # Context for current chunk
        context_for_current_chunk = (
            f"Initial User Request: {initial_user_prompt_for_session}\n\n"
            f"Overall Topic: {original_prompt}\n\n"
            f"Previously Generated Text (ensure continuity with this, this is the last part of it):\n"
            f"{all_generated_text[-10000:] if len(all_generated_text) > 10000 else all_generated_text}"
            f"{'... (truncated previous text)' if len(all_generated_text) > 10000 else ''}"
        )

        # Step 1: Gemini Draft
        gemini_draft_prompt = (
            f"{context_for_current_chunk}\n\n"
            f"Please write the next part of the text, aiming for approximately {target_chars_per_chunk} characters. "
            f"This is chunk {current_chunk_num} of {num_chunks}."
            "Focus on generating new content for the current chunk, building upon the previously generated text if available."
        )
        logger.info(f"Chunk {current_chunk_num} - Step 1: Gemini Draft")
        gemini_response = await get_gemini_response(
            request=request,
            prompt_text=gemini_draft_prompt,
            model_name="gemini-2.5-pro-preview-05-06", # Using specified model
            chat_history=chat_history_for_ai,
            initial_user_prompt=initial_user_prompt_for_session,
            user_memories=user_memories,
            # System instruction can be generic or specific if needed
            system_instruction="You are an AI assistant helping to draft a long text in chunks. Ensure continuity with previous text if provided."
        )
        intermediate_steps_details.append(gemini_response)

        if gemini_response.error or not gemini_response.response:
            logger.error(f"Chunk {current_chunk_num} - Gemini Draft failed or returned no response: {gemini_response.error}")
            # Decide on error handling: for now, log and use empty string, or potentially break/return.
            # response_shell.overall_error = f"Error in Gemini Draft for chunk {current_chunk_num}: {gemini_response.error}"
            # response_shell.ultra_writing_mode_details = intermediate_steps_details
            # return response_shell # Example of early exit
            current_chunk_text = "" # Fallback to empty if critical error
        else:
            current_chunk_text = gemini_response.response
            logger.info(f"Chunk {current_chunk_num} - Gemini Draft successful. Length: {len(current_chunk_text)}")


        # Step 2: Claude Refine
        if current_chunk_text: # Only refine if there's text from Gemini
            claude_refine_prompt = (
                f"{context_for_current_chunk}\n\n"
                f"AI-Generated Draft to Refine:\n{current_chunk_text}\n\n"
                f"Please refine this draft to be more engaging, well-structured, and stylistically appealing, "
                f"while ensuring the character count does not decrease significantly from the target of {target_chars_per_chunk} characters. "
                f"This is for chunk {current_chunk_num} of {num_chunks}."
            )
            logger.info(f"Chunk {current_chunk_num} - Step 2: Claude Refine")
            claude_response = await get_claude_response(
                request=request,
                prompt_text=claude_refine_prompt,
                model="claude-opus-4-20250514", # Using specified model
                chat_history=chat_history_for_ai,
                initial_user_prompt=initial_user_prompt_for_session,
                user_memories=user_memories,
                system_instruction="You are an AI assistant refining a draft. Focus on improving engagement, structure, and style."
            )
            intermediate_steps_details.append(claude_response)

            if claude_response.error or not claude_response.response:
                logger.warning(f"Chunk {current_chunk_num} - Claude Refine failed or returned no response: {claude_response.error}. Using text from Gemini draft.")
            else:
                current_chunk_text = claude_response.response
                logger.info(f"Chunk {current_chunk_num} - Claude Refine successful. Length: {len(current_chunk_text)}")
        else:
            logger.warning(f"Chunk {current_chunk_num} - Skipping Claude Refine as Gemini draft was empty.")


        # Step 3: Character Count Check & Expansion Loop
        logger.info(f"Chunk {current_chunk_num} - Step 3: Character Count Check & Expansion Loop")
        for i in range(max_retries_char_count):
            logger.info(f"Chunk {current_chunk_num} - Char Count Check Attempt {i+1}/{max_retries_char_count}")
            # Simple length check first (more reliable than asking AI)
            if len(current_chunk_text) >= target_chars_per_chunk * 0.8: # Allow 20% leeway
                 logger.info(f"Chunk {current_chunk_num} - Length {len(current_chunk_text)} is sufficient (>= 80% of {target_chars_per_chunk}).")
                 break # Length is fine

            # If still too short, ask Gemini to verify (as per plan, though direct check is better)
            gemini_check_length_prompt = (
                f"The following text is chunk {current_chunk_num} of {num_chunks}. "
                f"The target length is {target_chars_per_chunk} characters. Current text (length {len(current_chunk_text)}):\n{current_chunk_text}\n\n"
                f"Is the length of this text significantly less than {target_chars_per_chunk} characters? Answer with only 'YES' or 'NO'. "
                f"If it's reasonably close (e.g. 80% or more) or over, answer 'NO'."
            )
            gemini_length_check_response = await get_gemini_response(
                request=request,
                prompt_text=gemini_check_length_prompt,
                model_name="gemini-2.5-pro-preview-05-06",
                # Less context needed for simple check
            )
            intermediate_steps_details.append(gemini_length_check_response)

            if gemini_length_check_response.error:
                logger.warning(f"Chunk {current_chunk_num} - Gemini length check failed: {gemini_length_check_response.error}. Assuming length is OK to avoid loop error.")
                break

            is_too_short_str = gemini_length_check_response.response.strip().upper()
            logger.info(f"Chunk {current_chunk_num} - Gemini length check response: '{is_too_short_str}'")

            if "YES" in is_too_short_str or len(current_chunk_text) < target_chars_per_chunk * 0.5: # Add a hard threshold too
                logger.info(f"Chunk {current_chunk_num} - Text is too short (Gemini: {is_too_short_str}, Actual: {len(current_chunk_text)}). Attempting expansion with Claude.")
                claude_expand_prompt = (
                    f"{context_for_current_chunk}\n\n"
                    f"Previously Generated Text for this Chunk (too short, current length {len(current_chunk_text)}):\n{current_chunk_text}\n\n"
                    f"Please expand this text significantly to meet the target of approximately {target_chars_per_chunk} characters for chunk {current_chunk_num} of {num_chunks}. "
                    f"Add more details, examples, or elaborations as appropriate. Ensure the expansion is coherent and maintains quality."
                )
                claude_expand_response = await get_claude_response(
                    request=request,
                    prompt_text=claude_expand_prompt,
                    model="claude-opus-4-20250514",
                    system_instruction="You are an AI assistant expanding text to meet a target length. Add relevant details and ensure coherence.",
                    # Pass other relevant params like history, user_memories if needed for context
                    chat_history=chat_history_for_ai,
                    initial_user_prompt=initial_user_prompt_for_session,
                    user_memories=user_memories,
                )
                intermediate_steps_details.append(claude_expand_response)

                if claude_expand_response.error or not claude_expand_response.response:
                    logger.warning(f"Chunk {current_chunk_num} - Claude expansion failed: {claude_expand_response.error}. Using previous text.")
                    break # Break if expansion fails
                else:
                    current_chunk_text = claude_expand_response.response
                    logger.info(f"Chunk {current_chunk_num} - Claude expansion successful. New length: {len(current_chunk_text)}")
                    # Continue to re-check length in the next iteration of this loop
            else: # Gemini says 'NO' or length is fine
                logger.info(f"Chunk {current_chunk_num} - Length is considered sufficient by Gemini or direct check. Actual: {len(current_chunk_text)}.")
                break # Break from character count loop

        if len(current_chunk_text) < target_chars_per_chunk * 0.5: # Final check
             logger.warning(f"Chunk {current_chunk_num} - After expansion attempts, length {len(current_chunk_text)} is still significantly less than target {target_chars_per_chunk}.")


        # Step 4: Consistency Check & Fix Loop
        logger.info(f"Chunk {current_chunk_num} - Step 4: Consistency Check & Fix Loop")
        for i in range(max_retries_consistency):
            logger.info(f"Chunk {current_chunk_num} - Consistency Check Attempt {i+1}/{max_retries_consistency}")
            gemini_check_consistency_prompt = (
                f"Initial User Request: {initial_user_prompt_for_session}\n"
                f"Overall Topic: {original_prompt}\n"
                f"Previously Generated Text (ensure continuity with this, this is the last part of it):\n{all_generated_text[-10000:] if len(all_generated_text) > 10000 else all_generated_text}{'... (truncated previous text)' if len(all_generated_text) > 10000 else ''}\n\n"
                f"Current Chunk Draft (chunk {current_chunk_num} of {num_chunks}):\n{current_chunk_text}\n\n"
                f"Review the 'Current Chunk Draft'. Are there any plot holes, setting errors, character inconsistencies, or factual contradictions "
                f"when compared to the 'Previously Generated Text' or the 'Initial User Request' and 'Overall Topic'? "
                f"If major issues exist, describe them briefly (e.g., 'The character John was previously described as a doctor, but is now a pilot.'). "
                f"If no major issues, answer ONLY with the exact phrase 'NO MAJOR ISSUES'."
            )
            gemini_consistency_response = await get_gemini_response(
                request=request,
                prompt_text=gemini_check_consistency_prompt,
                model_name="gemini-2.5-pro-preview-05-06", # Powerful model for consistency
            )
            intermediate_steps_details.append(gemini_consistency_response)

            if gemini_consistency_response.error:
                logger.warning(f"Chunk {current_chunk_num} - Gemini consistency check failed: {gemini_consistency_response.error}. Assuming no major issues to avoid loop error.")
                break

            consistency_issues = gemini_consistency_response.response.strip()
            logger.info(f"Chunk {current_chunk_num} - Gemini consistency check response: '{consistency_issues}'")

            if consistency_issues != "NO MAJOR ISSUES":
                logger.info(f"Chunk {current_chunk_num} - Consistency issues found: {consistency_issues}. Attempting fix with Claude.")
                claude_fix_prompt = (
                    f"{context_for_current_chunk}\n\n" # context_for_current_chunk already has previous text snippet
                    f"Text of Current Chunk (chunk {current_chunk_num} of {num_chunks}):\n{current_chunk_text}\n\n"
                    f"An automated check found the following consistency issues with previously generated text or overall request: '{consistency_issues}'.\n"
                    f"Please revise the 'Text of Current Chunk' to address these issues while maintaining its core content and "
                    f"target length of approximately {target_chars_per_chunk} characters."
                )
                claude_fix_response = await get_claude_response(
                    request=request,
                    prompt_text=claude_fix_prompt,
                    model="claude-opus-4-20250514",
                    system_instruction="You are an AI assistant revising text to fix consistency issues. Ensure the revised text is coherent and addresses the identified problems.",
                    chat_history=chat_history_for_ai, # Provide history for broader context
                    initial_user_prompt=initial_user_prompt_for_session,
                    user_memories=user_memories,
                )
                intermediate_steps_details.append(claude_fix_response)

                if claude_fix_response.error or not claude_fix_response.response:
                    logger.warning(f"Chunk {current_chunk_num} - Claude consistency fix failed: {claude_fix_response.error}. Using previous text.")
                    break # Break if fix fails
                else:
                    old_len = len(current_chunk_text)
                    current_chunk_text = claude_fix_response.response
                    new_len = len(current_chunk_text)
                    logger.info(f"Chunk {current_chunk_num} - Claude consistency fix successful. Length changed from {old_len} to {new_len}.")
                    # As per spec, log length change but don't explicitly re-run length check loop here for simplicity.
                    # A more robust solution might re-trigger length check or integrate it.
                    # Continue to re-check consistency in the next iteration of this loop.
            else: # Gemini says 'NO MAJOR ISSUES'
                logger.info(f"Chunk {current_chunk_num} - No major consistency issues found.")
                break # Break from consistency loop

        # Store Finalized Chunk
        logger.info(f"Completed processing for chunk {current_chunk_num}/{num_chunks}. Final length: {len(current_chunk_text)}")
        chunk_data = schemas.IndividualAIResponse(
            source=f"Iterative Super Drafting - Chunk {current_chunk_num}/{num_chunks}",
            response=current_chunk_text,
            # error=None, # Assuming if we reach here, chunk is considered successful for now
        )
        completed_chunks_data.append(chunk_data)
        all_generated_text += current_chunk_text + "\n\n" # Separator for next iteration's context

    # Finalize and Return
    # The prompt asks for `final_combined_text = "".join(chunk.response for chunk in completed_chunks_data)`
    # However, `all_generated_text` already has them joined with "\n\n".
    # Using `all_generated_text` might be better if specific separators are desired.
    # For now, sticking to the explicit join as requested for the final shell, but `all_generated_text` was used for context.
    final_combined_text = ""
    for chunk in completed_chunks_data:
        if chunk.response: # Ensure only non-empty responses are joined
             final_combined_text += chunk.response + "\n\n" # Using \n\n as separator, can be adjusted

    # Trim trailing newlines if any
    final_combined_text = final_combined_text.strip()


    response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
        source="Iterative Super Drafting (Final)", response=final_combined_text
    )
    response_shell.ultra_writing_mode_details = intermediate_steps_details

    if not final_combined_text and not response_shell.overall_error : # If no text and no major error logged
        response_shell.overall_error = "Iterative Super Drafting completed but generated no content."
        logger.error("Iterative Super Drafting generated no final content.")
    elif not final_combined_text and response_shell.overall_error:
        logger.error(f"Iterative Super Drafting failed to generate content and error was: {response_shell.overall_error}")
    else:
        logger.info(f"Iterative Super Drafting flow completed. Total length: {len(final_combined_text)}")

    return response_shell
