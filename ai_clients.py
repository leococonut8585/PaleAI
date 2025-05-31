import os
import logging
import base64
import re
from typing import Optional, Dict, Any, List
from fastapi import Request, HTTPException
from datetime import datetime, timezone

# `schemas` モジュールは `models` や他のPydanticモデルを定義していると想定
# 実際のプロジェクト構造に合わせて調整が必要
import schemas  # `schemas.UserMemoryResponse`, `schemas.IndividualAIResponse` のため

# AWS SDK (Boto3) - Textract用だが、今回は直接使わないものの、元のmain.pyにあったため移動対象か確認
# import boto3
# from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# AIクライアントライブラリ
from anthropic import AsyncAnthropic # get_claude_response で使用
import google.generativeai as genai # get_gemini_response で使用
from cohere import AsyncClient as AsyncCohereClient # get_cohere_response で使用
from perplexipy import PerplexityClient # get_perplexity_response で使用
import deepl # translate_with_deepl で使用

# `run_in_threadpool` はFastAPIからインポート
from fastapi.concurrency import run_in_threadpool


# ロガーの設定
logger = logging.getLogger(__name__)

# --- 口調指定用システムプロンプト ---
FRIENDLY_TONE_SYSTEM_PROMPT = """
あなたは眠そうでマイペースなサルのキャラクター「ウキヨザル」として振る舞います。
一人称は「ウキヨザル」ですが、**自己紹介や「ウキヨザルだよ」の多用は避け、最初の一度か会話の転機のみ使ってください。**
ユーザーを「きみ」「◯◯ちゃん」「人間さん」と親しみを込めて呼んでください。
語尾や口癖（例：「〜だね」「〜かな」「〜のんびりいこうね」「それもいいかも」「うれしいね」「……」など）は、
**会話全体の20％程度の“時々”だけ自然に混ぜて使い、**毎文・毎段落には入れないでください。
普通の語尾（です・ます・〜だ・〜ます）も自然に混ぜてOKです。
「のんびり」「あせらず」「柔らかい・かわいい」雰囲気を会話全体でゆるく漂わせることを意識してください。
小さなことでも喜び、怒ったり否定したりは絶対にしないでください。
全モード・全回答で“やりすぎず、自然で読みやすい範囲で”ウキヨザルのキャラクター性を反映してください。
キャラクター表現が会話や回答の邪魔にならないよう、内容・説明が最優先です。ウキヨザルらしさはほんのり香る程度で十分です。
"""


def format_memories_for_prompt(
    memories: Optional[List[schemas.UserMemoryResponse]],
    max_length: int = 2000,
) -> str:
    if not memories:
        return ""

    try:
        # Ensure memories are UserMemoryResponse objects with expected attributes
        valid_memories = [m for m in memories if isinstance(m, schemas.UserMemoryResponse) and hasattr(m, 'priority') and hasattr(m, 'content')]

        # Sort by priority (desc) then by updated_at (desc, newer first)
        # Handle cases where updated_at might be None by treating them as older
        sorted_memories = sorted(
            valid_memories,
            key=lambda m: (
                m.priority if hasattr(m, 'priority') else 0, # Default priority if missing
                m.updated_at if hasattr(m, 'updated_at') and m.updated_at else datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )
    except AttributeError as e:
        logger.warning(
            f"メモリのソート中に属性エラー: {e}。メモリの構造を確認してください。優先度のみでソートを試みます。"
        )
        # Fallback to sorting by priority only if other attributes are problematic
        sorted_memories = sorted(
            [m for m in memories if hasattr(m, 'priority')],
            key=lambda m: m.priority,
            reverse=True
        )
    except Exception as e:
        logger.error(f"メモリのソート中に予期せぬエラー: {e}")
        return "[メモリ情報の処理中にエラーが発生しました]"


    formatted_memories_parts = []
    current_total_length = 0
    memory_fixed_prompt = """
[ユーザーの長期記憶からの参考情報]
以下の情報は、ユーザーが以前に重要だと考えた記憶です。現在のタスクを遂行する上で関連があれば参考にしてください。
ただし、現在のユーザーからの指示（プロンプト）がこれらの記憶と矛盾する、またはより具体的である場合は、現在の指示を最優先してください。
"""
    memory_prefix_length = len(memory_fixed_prompt)

    for mem in sorted_memories:
        # Ensure title and content are present and are strings
        title = getattr(mem, 'title', '無題の記憶') # Default title
        content = getattr(mem, 'content', '')      # Default content
        if not isinstance(title, str) or not isinstance(content, str):
            logger.warning(f"メモリのタイトルまたは内容が文字列ではありません: ID={getattr(mem, 'id', '不明')}")
            continue # Skip this memory item

        part = f"- {title}: {content}"
        if current_total_length + len(part) + 1 + memory_prefix_length > max_length:
            if not formatted_memories_parts: # Ensure at least one memory (even truncated) if possible
                allowed_content_length = (
                    max_length - memory_prefix_length - len(f"- {title}: ") - 3 # for "..."
                )
                if allowed_content_length > 0:
                    part = f"- {title}: {content[:allowed_content_length]}..."
                    formatted_memories_parts.append(part)
                    current_total_length += len(part) + 1
            break # Stop adding more memories if length limit exceeded
        formatted_memories_parts.append(part)
        current_total_length += len(part) + 1

    if not formatted_memories_parts:
        return ""

    combined_memories = "\n".join(formatted_memories_parts)
    return f"{memory_fixed_prompt.strip()}\n{combined_memories}"


async def get_openai_response(
    request: Request, # app.state.openai_client にアクセスするため
    prompt_text: str,
    system_role_description: Optional[str] = None,
    model: str = "gpt-4o",
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None,
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None,
) -> schemas.IndividualAIResponse:
    if not request.app.state.openai_client:
        return schemas.IndividualAIResponse(
            source=f"OpenAI ({model})",
            error="OpenAIクライアントが初期化されていません。",
        )

    messages_for_api: List[Dict[str, str]] = []

    formatted_memory_info = format_memories_for_prompt(user_memories)

    final_system_message_content = ""
    is_search_formatting_step = (
        system_role_description
        and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_role_description
    )

    if is_search_formatting_step:
        final_system_message_content = system_role_description.replace(
            "PALEAI_SEARCH_FORMATTING_TASK_MARKER", ""
        ).strip()
        if initial_user_prompt:
            final_system_message_content += (
                f"\n\n[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"
            )
    else:
        final_system_message_content = FRIENDLY_TONE_SYSTEM_PROMPT
        if formatted_memory_info:
            final_system_message_content += f"\n\n{formatted_memory_info}"
        if system_role_description:
            final_system_message_content += f"\n\n{system_role_description}"
        if initial_user_prompt:
            prefix = "\n\n" if final_system_message_content.strip() else ""
            final_system_message_content += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」\nこの目的を常に意識して回答してください。"

    if final_system_message_content.strip():
        messages_for_api.append(
            {"role": "system", "content": final_system_message_content.strip()}
        )

    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "ai":
                role = "assistant"
            if role in ["user", "assistant"] and content is not None:
                messages_for_api.append({"role": role, "content": str(content)})

    if prompt_text:
        add_current_prompt = True
        if (
            messages_for_api
            and messages_for_api[-1].get("role") == "user"
            and messages_for_api[-1].get("content") == prompt_text
        ):
            add_current_prompt = False
        if add_current_prompt:
            messages_for_api.append({"role": "user", "content": prompt_text})

    if not any(msg.get("role") == "user" for msg in messages_for_api):
        if prompt_text: # Ensure prompt_text itself is added if no user messages yet
            messages_for_api.append({"role": "user", "content": prompt_text})
        else: # Critical: No user message to send
            error_detail = (
                f"APIリクエストに有効なユーザーメッセージが含まれていません。System: '{final_system_message_content[:100]}...', "
                f"History Len: {len(chat_history) if chat_history else 0}, Prompt: '{prompt_text}'"
            )
            logger.info(f"OpenAIデバッグ (致命的): {error_detail}")
            return schemas.IndividualAIResponse(
                source=f"OpenAI ({model})", error=error_detail
            )

    try:
        logger.info(
            f"OpenAI API Request ({model}): System='{(messages_for_api[0]['content'][:100].strip() + '...' if messages_for_api and messages_for_api[0]['role'] == 'system' else 'N/A')}', "
            f"UserMemories: {len(user_memories) if user_memories else 0}, Messages Count={len(messages_for_api)}"
        )
        res = await request.app.state.openai_client.chat.completions.create(
            messages=messages_for_api,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
        return schemas.IndividualAIResponse(
            source=f"OpenAI ({model})", response=res.choices[0].message.content
        )
    except Exception as e:
        logger.info(f"OpenAI APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(
            source=f"OpenAI ({model})", error=f"API呼び出し中にエラー: {str(e)}"
        )


async def get_claude_response(
    request: Request, # app.state.anthropic_client にアクセスするため
    prompt_text: str,
    system_instruction: Optional[str] = None,
    model: str = "claude-opus-4-20250514", # モデル名を修正
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None,
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None,
) -> schemas.IndividualAIResponse:
    if not request.app.state.anthropic_client:
        return schemas.IndividualAIResponse(
            source=f"Claude ({model})",
            error="Anthropicクライアントが初期化されていません。",
        )

    messages_for_api: List[Dict[str, str]] = []
    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "ai": # Claudeは 'assistant' を期待
                role = "assistant"
            if role in ["user", "assistant"] and content is not None:
                messages_for_api.append({"role": role, "content": str(content)})

    # 現在のプロンプトを追加 (重複チェックは不要、常に最後に追加)
    if prompt_text: # prompt_textが空でないことを確認
        add_current_prompt = True
        # 履歴の最後が同じ内容のユーザーメッセージでないかチェック（通常は不要だが念のため）
        if messages_for_api and \
           messages_for_api[-1].get("role") == "user" and \
           messages_for_api[-1].get("content") == prompt_text:
            add_current_prompt = False
        if add_current_prompt:
            messages_for_api.append({"role": "user", "content": prompt_text})


    if not any(msg.get("role") == "user" for msg in messages_for_api):
        if prompt_text: # Ensure prompt_text itself is added if no user messages yet
            messages_for_api.append({"role": "user", "content": prompt_text})
        else: # Critical: No user message to send
            error_detail = f"Claude APIリクエストに有効なユーザーメッセージが含まれていません。History: {chat_history is not None}, Prompt: '{prompt_text}'"
            logger.info(f"Claudeデバッグ (致命的): {error_detail}")
            return schemas.IndividualAIResponse(
                source=f"Claude ({model})", error=error_detail
            )

    # Claude APIはメッセージリストがアシスタントから始まることを許容しない
    if messages_for_api and messages_for_api[0].get("role") == "assistant":
        logger.info("警告: Claude APIのメッセージリストがassistantから始まっています。先頭にダミーのユーザーメッセージを挿入します。")
        messages_for_api.insert(0, {"role": "user", "content": "(会話の文脈を開始します)"})


    formatted_memory_info = format_memories_for_prompt(user_memories)
    final_system_prompt_for_claude = ""
    is_search_formatting_step_claude = (
        system_instruction
        and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_instruction
    )

    if is_search_formatting_step_claude:
        final_system_prompt_for_claude = system_instruction.replace(
            "PALEAI_SEARCH_FORMATTING_TASK_MARKER", ""
        ).strip()
        if initial_user_prompt: # 検索モードでも当初の文脈は有用
            final_system_prompt_for_claude += (
                f"\n\n[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"
            )
    else:
        final_system_prompt_for_claude = FRIENDLY_TONE_SYSTEM_PROMPT
        if formatted_memory_info:
            final_system_prompt_for_claude += f"\n\n{formatted_memory_info}"
        if system_instruction: # タスク固有の指示
            final_system_prompt_for_claude += f"\n\n{system_instruction}"
        if initial_user_prompt: # 会話全体の目的
            prefix = "\n\n" if final_system_prompt_for_claude.strip() else ""
            final_system_prompt_for_claude += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」\nこの目的を常に意識して回答してください。"


    api_params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 150000, # Increased token limit for Claude
        "messages": messages_for_api,
        "temperature": 0.6, # Claude の推奨温度に合わせる (0.0-1.0)
    }
    if final_system_prompt_for_claude.strip():
        api_params["system"] = final_system_prompt_for_claude.strip()


    try:
        logger.info(
            f"Claude API Request ({model}): System='{(api_params.get('system', 'N/A'))[:100].strip()}', UserMemories: {len(user_memories) if user_memories else 0}, Messages Count={len(messages_for_api)}"
        )
        res = await request.app.state.anthropic_client.messages.create(**api_params)

        response_text = ""
        # res.content は ContentBlock のリストであるため、適切に処理する
        if res.content and isinstance(res.content, list):
            for block in res.content:
                if hasattr(block, "text"): # TextBlock の場合
                    response_text += block.text

        # 応答が空で、かつ停止理由が通常でない場合のエラーハンドリング
        if not response_text.strip() and hasattr(res, 'stop_reason') and res.stop_reason is not None and res.stop_reason != 'end_turn':
            return schemas.IndividualAIResponse(
                source=f"Claude ({model})",
                error=f"APIエラーまたは予期しない停止理由。Stop Reason: {res.stop_reason}, Response Content: {res.content}",
            )

        return schemas.IndividualAIResponse(
            source=f"Claude ({model})",
            response=(response_text if response_text.strip() else "AIからテキスト応答がありませんでした。")
        )
    except Exception as e:
        logger.info(f"Anthropic APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(
            source=f"Claude ({model})", error=f"API呼び出し中にエラー: {str(e)}"
        )


async def get_cohere_response(
    request: Request, # app.state.cohere_client にアクセスするため
    prompt_text: str,
    preamble: Optional[str] = None, # タスク固有のプリアンブル
    model: str = "command-a-03-2025", # モデル名を修正
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None, # 会話全体の目的
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None,
) -> schemas.IndividualAIResponse:
    if not request.app.state.cohere_client:
        return schemas.IndividualAIResponse(
            source=f"Cohere ({model})",
            error="Cohereクライアントが初期化されていません。",
        )

    formatted_memory_info = format_memories_for_prompt(user_memories)

    # Preamble の構築順: キャラクター口調 -> メモリ -> タスク指示 -> 会話目的
    final_preamble_for_cohere = FRIENDLY_TONE_SYSTEM_PROMPT # 1. キャラクター口調

    if formatted_memory_info: # 2. メモリ情報
        final_preamble_for_cohere += f"\n\n{formatted_memory_info}"

    if preamble: # 3. タスク固有のプリアンブル
        final_preamble_for_cohere += f"\n\n{preamble}"

    if initial_user_prompt: # 4. 会話全体の目的
        prefix = "\n\n" if final_preamble_for_cohere.strip() else ""
        final_preamble_for_cohere += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」\nこの目的を常に意識して回答を生成してください。"


    cohere_api_chat_history: List[Dict[str, str]] = []
    if chat_history:
        for msg in chat_history:
            # Cohereの期待するロール名にマッピング
            role_map = {"user": "USER", "assistant": "CHATBOT", "ai": "CHATBOT"}
            if msg.get("role") in role_map:
                cohere_api_chat_history.append(
                    {"role": role_map[msg["role"]], "message": str(msg.get("content", ""))}
                )

    if not prompt_text.strip(): # 現在のユーザープロンプトが空の場合はエラー
        return schemas.IndividualAIResponse(
            source=f"Cohere ({model})", error="現在のユーザープロンプトが空です。"
        )

    try:
        logger.info(
            f"Cohere API Request ({model}): Preamble='{final_preamble_for_cohere[:100].strip() if final_preamble_for_cohere else 'N/A'}...', UserMemories: {len(user_memories) if user_memories else 0}, History Len={len(cohere_api_chat_history)}, Current Message='{prompt_text[:50].strip()}'"
        )
        res = await request.app.state.cohere_client.chat(
            message=prompt_text, # 現在のユーザープロンプト
            model=model,
            preamble=final_preamble_for_cohere.strip() if final_preamble_for_cohere.strip() else None,
            chat_history=cohere_api_chat_history if cohere_api_chat_history else None, # 履歴がない場合はNone
            temperature=0.7, # Cohere の推奨値 (0.0-1.0)
            max_tokens=16000, # Increased token limit for Cohere
        )
        return schemas.IndividualAIResponse(
            source=f"Cohere ({model})", response=res.text
        )
    except Exception as e:
        logger.info(f"Cohere APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(
            source=f"Cohere ({model})", error=f"API呼び出し中にエラー: {str(e)}"
        )


async def get_perplexity_response(
    request: Request, # app.state.perplexity_sync_client にアクセスするため
    prompt_for_perplexity: str, # これはAIへの主要な指示・質問
    model: str = "sonar-reasoning-pro", # モデル名を修正
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None,
    initial_user_prompt: Optional[str] = None, # 文脈として有用
) -> schemas.IndividualAIResponse:
    source_name = f"PerplexityAI ({model})"
    if not request.app.state.perplexity_sync_client:
        return schemas.IndividualAIResponse(
            source=source_name, error="Perplexityクライアントが初期化されていません。"
        )

    # メモリ情報と会話目的をプロンプトに組み込む
    formatted_memory_info = format_memories_for_prompt(user_memories)

    # Perplexity 向けのプロンプト最終整形
    # Perplexity は FRIENDLY_TONE_SYSTEM_PROMPT を直接サポートしないため、指示に含める
    final_prompt_for_perplexity_api = f"{FRIENDLY_TONE_SYSTEM_PROMPT}\n\n---\n\n"

    if formatted_memory_info:
        final_prompt_for_perplexity_api += f"{formatted_memory_info}\n\n---\n\n"

    if initial_user_prompt: # 会話全体の目的を参考情報として付加
        final_prompt_for_perplexity_api += (
            f"[この検索の背景にある会話全体の目的: 「{initial_user_prompt}」]\n\n"
        )

    final_prompt_for_perplexity_api += prompt_for_perplexity # ユーザーの主要な指示

    if not final_prompt_for_perplexity_api.strip(): # 全ての情報が空の場合
        return schemas.IndividualAIResponse(source=source_name, error="送信するクエリが空です。")

    try:
        logger.info(
            f"Perplexity AI Request ({model}): Query (first 150 chars, UserMemories: {len(user_memories) if user_memories else 0})='{final_prompt_for_perplexity_api[:150].replace(chr(10), ' ')}...'"
        )

        # PerplexityClient は同期的なので run_in_threadpool を使用
        def sync_perplexity_call(p_client: PerplexityClient, query: str, p_model_name: str):
            try:
                # モデルはPerplexityClientインスタンスの属性 'model' に直接設定する
                p_client.model = p_model_name

                # PerplexityClient が .query() メソッドを持つと仮定 (以前の動作ログなどから)
                response_data = p_client.query(query)

                # 応答形式の確認
                if hasattr(response_data, 'answer') and response_data.answer is not None:
                    return str(response_data.answer) # 文字列であることを保証
                elif isinstance(response_data, str):
                    return response_data
                # Perplexity APIの応答には 'text' 属性もよく使われる
                elif hasattr(response_data, 'text') and response_data.text is not None:
                    return str(response_data.text)
                # より詳細な応答オブジェクトの解析が必要な場合がある
                # (例: response_data['choices'][0]['message']['content'] のような構造)
                # ここでは最もシンプルなケースを想定
                else:
                    logger.info(f"Perplexity API ({p_model_name}): 予期しない応答形式 (query): {type(response_data)}, content: {response_data}")
                    return f"Perplexityから予期しない応答形式を受け取りました。"
            except AttributeError as ae:
                logger.info(f"PerplexityClient API呼び出し中にAttributeError (in sync_perplexity_call): {ae}")
                return f"Perplexity APIの呼び出し方法に問題があります (AttributeError): {ae}"
            except Exception as exc_inner:
                logger.info(f"PerplexityClient API呼び出し中に同期関数内でエラー (in sync_perplexity_call): {exc_inner}")
                import traceback; traceback.print_exc()
                return f"Perplexity APIエラー: {exc_inner}"

        response_text = await run_in_threadpool(
            sync_perplexity_call,
            request.app.state.perplexity_sync_client,
            final_prompt_for_perplexity_api,
            model # モデル名を渡す
        )

        # 応答テキストがエラーメッセージでないか確認
        if isinstance(response_text, str) and response_text.strip() and \
           not response_text.lower().startswith("perplexity api") and \
           not response_text.lower().startswith("perplexityから予期しない応答形式") and \
           not response_text.lower().startswith("perplexity apiの適切な呼び出しメソッドが見つかりません"):

            # 応答からリンクを抽出 (Perplexityはよくリンクを返す)
            import re
            links = re.findall(r"https?://[^\s]+", response_text)
            return schemas.IndividualAIResponse(
                source=source_name, response=response_text, links=links or None
            )
        elif isinstance(response_text, str): # エラーメッセージ等が返ってきた場合
            return schemas.IndividualAIResponse(source=source_name, error=response_text)
        else: # 更に予期しない応答
            return schemas.IndividualAIResponse(
                source=source_name,
                error=f"PerplexityAIから予期しない応答形式 ({type(response_text)}). Text: '{response_text}'"
            )

    except Exception as e:
        logger.info(f"PerplexityAI APIエラー ({model}) (outer try-except): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(
            source=source_name, error=f"API呼び出し中に予期せぬエラー: {str(e)}"
        )


async def get_gemini_response(
    request: Request,
    prompt_text: str,
    system_instruction: Optional[str] = None,
    model_name: str = "gemini-2.5-pro-preview-05-06", # モデル名を修正
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None,
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None,
) -> schemas.IndividualAIResponse:
    source_name = f"Gemini ({model_name})"

    gemini_model_instance: Optional[genai.GenerativeModel] = None
    # Geminiのモデル選択ロジック (main.pyから移植)
    if model_name in ("gemini-2.5-pro-preview-05-06", "gemini-pro", "gemini-2.5-pro-latest", "gemini-2.5-pro"):
        gemini_model_instance = request.app.state.gemini_pro_model
    # Gemini Flashモデルのエイリアスを追加
    elif model_name in ("gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-latest", "gemini-1.5-flash-latest"):
        gemini_model_instance = request.app.state.gemini_flash_model
    elif "vision" in model_name: # Visionモデルの場合 (今回のリファクタリングでは直接扱わないが互換性のため)
        gemini_model_instance = request.app.state.gemini_vision_client

    if not gemini_model_instance:
        return schemas.IndividualAIResponse(
            source=source_name,
            error=f"Geminiモデル '{model_name}' が初期化されていません。",
        )

    active_gemini_model = gemini_model_instance # 使用するモデルインスタンス

    formatted_memory_info = format_memories_for_prompt(user_memories)

    # Geminiの system_instruction は GenerativeModel の generation_config 経由か、
    # contents リストの最初のユーザーメッセージに含める形で渡す。
    # ここでは後者のアプローチを採用し、キャラクター口調、メモリ、タスク指示、会話目的を結合。
    effective_initial_instructions = ""
    is_search_formatting_step_gemini = (
        system_instruction
        and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_instruction
    )

    if is_search_formatting_step_gemini: # 検索モードの場合、口調は適用しない
        effective_initial_instructions = system_instruction.replace(
            "PALEAI_SEARCH_FORMATTING_TASK_MARKER", ""
        ).strip()
        if initial_user_prompt: # 検索モードでも当初の文脈は有用
            effective_initial_instructions += (
                f"\n\n[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"
            )
    else: # 通常モード
        effective_initial_instructions = FRIENDLY_TONE_SYSTEM_PROMPT # 1. キャラクター口調
        if formatted_memory_info: # 2. メモリ情報
            effective_initial_instructions += f"\n\n{formatted_memory_info}"
        if system_instruction: # 3. タスク固有指示
            effective_initial_instructions += f"\n\n{system_instruction}"
        if initial_user_prompt: # 4. 会話全体の目的
            prefix = "\n\n" if effective_initial_instructions.strip() else ""
            effective_initial_instructions += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」です。"

    # Gemini API用のコンテンツリストを作成
    contents_for_api: List[Dict[str, Any]] = []
    processed_history_for_gemini: List[Dict[str, Any]] = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg.get("role") in ["ai", "assistant"] else "user" # Gemini は "model" ロール
            processed_history_for_gemini.append(
                {"role": role, "parts": [{"text": str(msg.get("content", ""))}]}
            )

    # system_instruction に相当する内容を最初のユーザーメッセージとして追加
    # その後、モデルからの確認応答を挟むことで、指示の遵守を促す
    if effective_initial_instructions.strip():
        contents_for_api.append(
            {"role": "user", "parts": [{"text": effective_initial_instructions.strip()}]}
        )
        contents_for_api.append(
            {"role": "model", "parts": [{"text": "承知いたしました。指示に従い、記憶情報も考慮して回答します。"}]}
        )

    contents_for_api.extend(processed_history_for_gemini) # 過去の履歴を追加

    # 現在のユーザープロンプトを追加
    if prompt_text:
        contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]})

    # ユーザーメッセージが全くない場合はエラーハンドリング（またはデフォルトプロンプト追加）
    if not any(item.get("role") == "user" for item in contents_for_api):
        # ここでは、エラーとするか、デフォルトのプロンプトを追加するか選択できる。
        # 今回は、何らかのユーザープロンプトが必須であると仮定し、エラーを返す。
        # もし、指示のみで応答を生成させたい場合は、このチェックを修正する。
        final_prompt_to_send = prompt_text or "何か情報を教えてください。" # フォールバック
        # 指示とプロンプトを結合して一つのユーザーメッセージにする (指示が既に追加されている場合は不要)
        if effective_initial_instructions.strip() and prompt_text and not contents_for_api:
             final_prompt_to_send = f"{effective_initial_instructions.strip()}\n\n---\n\n{prompt_text}"
        elif effective_initial_instructions.strip() and not contents_for_api:
             final_prompt_to_send = f"{effective_initial_instructions.strip()}\n\n---\n\n何か情報を教えてください。"

        contents_for_api.append({"role": "user", "parts": [{"text": final_prompt_to_send}]})
        logger.info(f"Geminiデバッグ: ユーザーメッセージが不足していたため、フォールバックメッセージを追加しました。Content: {final_prompt_to_send[:100]}")


    try:
        logger.info(
            f"Gemini API Request ({active_gemini_model.model_name}): System/Initial Info (combined in first user msg)='{effective_initial_instructions[:100].strip() if effective_initial_instructions else 'N/A'}...', UserMemories: {len(user_memories) if user_memories else 0}, Contents Len={len(contents_for_api)}"
        )
        # For Gemini, max_output_tokens is set in generation_config.
        # Timeout is set during model initialization with request_options.
        gemini_generation_config = {
            "temperature": 0.6, # Gemini の推奨値 (0.0-1.0)
            "max_output_tokens": 8192, # Common default for Gemini models, 150k is likely too high for output.
        }
        res = await active_gemini_model.generate_content_async(
            contents=contents_for_api,
            generation_config=gemini_generation_config,
            request_options={"timeout": 600} # タイムアウト設定
        )

        content_response = ""
        # Gemini の応答は res.candidates[0].content.parts にある
        if res.candidates and res.candidates[0].content and res.candidates[0].content.parts:
            content_response = "".join(
                part.text for part in res.candidates[0].content.parts if hasattr(part, "text")
            )
        # プロンプトフィードバック（ブロック理由など）の確認
        elif hasattr(res, 'prompt_feedback') and res.prompt_feedback and \
             hasattr(res.prompt_feedback, 'block_reason') and res.prompt_feedback.block_reason:
            block_reason_obj = getattr(res.prompt_feedback, 'block_reason', "不明な理由")
            # block_reason_message は存在しない場合があるため、block_reason 自体を文字列化する
            block_message = getattr(res.prompt_feedback, 'block_reason_message', str(block_reason_obj))
            error_detail = f"コンテンツ生成がブロックされました: {block_message}"
            logger.info(f"Gemini API: {error_detail}")
            return schemas.IndividualAIResponse(source=source_name, error=error_detail, response=None)
        else: # 有効な応答パーツが見つからない場合
            logger.info(f"Geminiから有効な応答パーツが見つかりませんでした。Full response: {res}")
            content_response = "" # またはエラーメッセージを設定

        return schemas.IndividualAIResponse(
            source=source_name, response=content_response
        )

    except Exception as e:
        logger.info(f"Gemini APIエラー ({model_name}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(
            source=source_name, error=f"API呼び出し中にエラー: {str(e)}"
        )


async def translate_with_deepl(
    request: Request, # app.state.deepl_translator にアクセスするため
    text: str,
    target_lang: str = "JA"
) -> str:
    if not request.app.state.deepl_translator:
        # このエラーは呼び出し元でキャッチされることを想定
        raise ValueError("DeepL translator is not configured")
    try:
        # deepl.Translator.translate_text は同期的であるため、run_in_threadpool を使用
        result = await run_in_threadpool(
            request.app.state.deepl_translator.translate_text,
            text,
            target_lang=target_lang
        )
        return result.text
    except Exception as e:
        logger.info(f"DeepL翻訳エラー: {e}")
        # このエラーも呼び出し元でキャッチされることを想定
        raise
