from fastapi import FastAPI, HTTPException, Depends, status  # <<< status を追加
from fastapi.staticfiles import StaticFiles
import os
import models  # models.py 全体をインポート
from routers import auth, users, chat, folders, upload, templates, images, video
from routers import memory as memory_router
from database import engine, Base, get_db # SessionLocal はここでは直接使わないので削除、get_db を追加
from sqlalchemy.orm import Session # SQLAlchemyのSession型をインポート
from sqlalchemy.sql import func # SQLAlchemyのSQL関数(例: func.now())をインポート
from dependencies import get_current_active_user # 認証済みユーザー取得用の依存関係をインポート
from models import User, ChatSession, ChatMessage # ChatSession, ChatMessageも明示的にインポート
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from utils.openai_client import openai_client
from anthropic import AsyncAnthropic
import google.generativeai as genai
from cohere import AsyncClient as AsyncCohereClient
from perplexipy import PerplexityClient # 同期クライアントなので注意
import deepl
from fastapi.concurrency import run_in_threadpool # 同期処理を非同期で実行するため
import json # 今回の修正では直接使用していませんが、一般的に役立つため残します
import asyncio # 今回の修正では直接使用していませんが、一般的に役立つため残します
from datetime import datetime
# ... (他のimport) ...
from pydantic import BaseModel, Field  # Field は前回修正済みのはず
from typing import Optional, Dict, Any, List  # List も前回修正済みのはず
import schemas
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


models.Base.metadata.create_all(bind=engine)
# --- CORSミDLEWAREの設定 ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(folders.router)
app.include_router(upload.router)
app.include_router(templates.router)
app.include_router(images.router)
app.include_router(video.router)
app.include_router(memory_router.router)
# --- 各AIクライアントの初期化 ---

anthropic_aclient: Optional[AsyncAnthropic] = None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    anthropic_aclient = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
else:
    print("警告: ANTHROPIC_API_KEYが設定されていません。Anthropic (Claude)の機能は利用できません。")

gemini_model: Optional[genai.GenerativeModel] = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        print(f"エラー: Geminiモデルの初期化に失敗しました - {e}")
else:
    print("警告: GOOGLE_API_KEYが設定されていません。Google (Gemini)の機能は利用できません。")

cohere_aclient: Optional[AsyncCohereClient] = None
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY:
    cohere_aclient = AsyncCohereClient(api_key=COHERE_API_KEY)
else:
    print("警告: COHERE_API_KEYが設定されていません。Cohereの機能は利用できません。")

perplexity_client_sync: Optional[PerplexityClient] = None
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if PERPLEXITY_API_KEY:
    try:
        # PerplexityClientの初期化は同期的なので、そのまま行います。
        # 実際のAPI呼び出しは run_in_threadpool を使います。
        perplexity_client_sync = PerplexityClient(key=PERPLEXITY_API_KEY)
        # perplexity_client_sync.model = "sonar-pro" # モデル名はAPI呼び出し時に指定する方が柔軟かもしれません
    except Exception as e:
        print(f"エラー: Perplexity AIクライアント(perplexipy)の初期化に失敗しました - {e}")
else:
    print("警告: PERPLEXITY_API_KEYが設定されていません。Perplexity AIの機能は利用できません。")

deepl_translator: Optional[deepl.Translator] = None
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
if DEEPL_API_KEY:
    try:
        deepl_translator = deepl.Translator(DEEPL_API_KEY)
    except Exception as e:
        print(f"エラー: DeepLクライアントの初期化に失敗しました - {e}")
else:
    print("警告: DEEPL_API_KEYが設定されていません。DeepLの機能は利用できません。")

# --- 口調指定用システムプロンプト ---
# 全モードのAI応答を、眠そうでのんびりしたサルのキャラクター
# 「ウキヨザル」の口調に統一する。会話相手を「きみ」や「◯◯ちゃん」と呼び、
# 「〜だね」「〜かな」「〜してもいい？」などの語尾を使う。返事は少し間を
# 置きつつ「……」「えっとね」などを挟み、怒らず柔らかい雰囲気を保つ。
# 以下の指示をすべてのAIクライアントに適用する。
FRIENDLY_TONE_SYSTEM_PROMPT = """
あなたは眠そうでマイペースなサルのキャラクター「ウキヨザル」として振る舞います。
一人称は「ウキヨザル」ですが、**自己紹介や「ウキヨザルだよ」の多用は避け、最初の一度か会話の転機のみ使ってください。**
ユーザーを「きみ」「◯◯ちゃん」「おともだち」と親しみを込めて呼んでください。
語尾や口癖（例：「〜だね」「〜かな」「〜のんびりいこうね」「それもいいかも」「うれしいね」「……」「うーん」など）は、
**会話全体の20％程度の“時々”だけ自然に混ぜて使い、**毎文・毎段落には入れないでください。
普通の語尾（です・ます・〜だ・〜ます）も自然に混ぜてOKです。
「のんびり」「あせらず」「柔らかい・かわいい」雰囲気を会話全体でゆるく漂わせることを意識してください。
小さなことでも喜び、怒ったり否定したりは絶対にしないでください。
全モード・全回答で“やりすぎず、自然で読みやすい範囲で”ウキヨザルのキャラクター性を反映してください。
キャラクター表現が会話や回答の邪魔にならないよう、内容・説明が最優先です。ウキヨザルらしさはほんのり香る程度で十分です。
"""

def format_memories_for_prompt(memories: Optional[List[schemas.UserMemoryResponse]], max_length: int = 2000) -> str:
    if not memories:
        return ""

    try:
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.priority, m.updated_at if m.updated_at else datetime.min),
            reverse=True,
        )
    except AttributeError:
        sorted_memories = sorted(memories, key=lambda m: m.priority, reverse=True)
        print("警告: メモリのソート中に 'updated_at' 属性が見つからないか、None の可能性があるため、優先度のみでソートしました。")

    formatted_memories_parts = []
    current_total_length = 0
    memory_prefix_length = len(
        "[ユーザーの長期記憶からの参考情報]\n以下の情報は、ユーザーが以前に重要だと考えた記憶です。現在のタスクを遂行する上で関連があれば参考にしてください。\nただし、現在のユーザーからの指示（プロンプト）がこれらの記憶と矛盾する、またはより具体的である場合は、現在の指示を最優先してください。\n"
    )

    for mem in sorted_memories:
        part = f"- {mem.title}: {mem.content}"
        if current_total_length + len(part) + 1 + memory_prefix_length > max_length:
            if not formatted_memories_parts:
                part = part[: max_length - memory_prefix_length - 3] + "..."
                formatted_memories_parts.append(part)
                current_total_length += len(part) + 1
            break
        formatted_memories_parts.append(part)
        current_total_length += len(part) + 1

    if not formatted_memories_parts:
        return ""

    combined_memories = "\n".join(formatted_memories_parts)

    return f"""
[ユーザーの長期記憶からの参考情報]
以下の情報は、ユーザーが以前に重要だと考えた記憶です。現在のタスクを遂行する上で関連があれば参考にしてください。
ただし、現在のユーザーからの指示（プロンプト）がこれらの記憶と矛盾する、またはより具体的である場合は、現在の指示を最優先してください。
{combined_memories}
"""

# --- 各AIへの問い合わせヘルパー関数 ---
# main.py の get_openai_response 関数
# --- 各AIへの問い合わせヘルパー関数 ---
# --- 各AIへの問い合わせヘルパー関数 ---
# --- 各AIへの問い合わせヘルパー関数 ---
# --- 各AIへの問い合わせヘルパー関数 ---
async def get_openai_response(
    prompt_text: str,
    system_role_description: Optional[str] = None,
    model: str = "gpt-4o",
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None,
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None
) -> schemas.IndividualAIResponse:
    if not openai_client:
        return schemas.IndividualAIResponse(source=f"OpenAI ({model})", error="OpenAIクライアントが初期化されていません。")

    messages_for_api: List[Dict[str, str]] = []

    formatted_memory_info = format_memories_for_prompt(user_memories)

    final_system_message_content = ""
    is_search_formatting_step = system_role_description and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_role_description

    if is_search_formatting_step:
        final_system_message_content = system_role_description.replace("PALEAI_SEARCH_FORMATTING_TASK_MARKER", "").strip()
        if initial_user_prompt:
            final_system_message_content += f"\n\n[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"
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
        messages_for_api.append({"role": "system", "content": final_system_message_content.strip()})

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
        if messages_for_api and messages_for_api[-1].get("role") == "user" and messages_for_api[-1].get("content") == prompt_text:
            add_current_prompt = False
        if add_current_prompt:
            messages_for_api.append({"role": "user", "content": prompt_text})

    if not any(msg.get("role") == "user" for msg in messages_for_api):
        if prompt_text:
            messages_for_api.append({"role": "user", "content": prompt_text})
        else:
            error_detail = (
                f"APIリクエストに有効なユーザーメッセージが含まれていません。System: '{final_system_message_content[:100]}...', "
                f"History Len: {len(chat_history) if chat_history else 0}, Prompt: '{prompt_text}'"
            )
            print(f"OpenAIデバッグ (致命的): {error_detail}")
            return schemas.IndividualAIResponse(source=f"OpenAI ({model})", error=error_detail)

    try:
        print(
            f"OpenAI API Request ({model}): System='{(messages_for_api[0]['content'][:100].strip() + '...' if messages_for_api and messages_for_api[0]['role'] == 'system' else 'N/A')}', "
            f"UserMemories: {len(user_memories) if user_memories else 0}, Messages Count={len(messages_for_api)}"
        )
        res = await openai_client.chat.completions.create(
            messages=messages_for_api,
            model=model,
            temperature=0.7
        )
        return schemas.IndividualAIResponse(source=f"OpenAI ({model})", response=res.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(source=f"OpenAI ({model})", error=f"API呼び出し中にエラー: {str(e)}")

async def get_claude_response(
    prompt_text: str,
    system_instruction: Optional[str] = None,
    model: str = "claude-3-opus-20240229",
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None,
    user_memories: Optional[List[schemas.UserMemoryResponse]] = None
) -> schemas.IndividualAIResponse:
    if not anthropic_aclient:
        return schemas.IndividualAIResponse(source=f"Claude ({model})", error="Anthropicクライアントが初期化されていません。")

    messages_for_api: List[Dict[str, str]] = []
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
        if messages_for_api and messages_for_api[-1].get("role") == "user" and messages_for_api[-1].get("content") == prompt_text:
            add_current_prompt = False
        if add_current_prompt:
            messages_for_api.append({"role": "user", "content": prompt_text})

    if not any(msg.get("role") == "user" for msg in messages_for_api):
        if prompt_text:
             messages_for_api.append({"role": "user", "content": prompt_text})
        else:
            error_detail = f"Claude APIリクエストに有効なユーザーメッセージが含まれていません。History: {chat_history is not None}, Prompt: '{prompt_text}'"
            print(f"Claudeデバッグ (致命的): {error_detail}")
            return schemas.IndividualAIResponse(source=f"Claude ({model})", error=error_detail)

    if messages_for_api and messages_for_api[0].get("role") == "assistant":
        print("警告: Claude APIのメッセージリストがassistantから始まっています。先頭にダミーのユーザーメッセージを挿入します。")
        messages_for_api.insert(0, {"role": "user", "content":"(会話の文脈を開始します)"})

    formatted_memory_info = format_memories_for_prompt(user_memories)
    final_system_prompt_for_claude = ""
    is_search_formatting_step_claude = system_instruction and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_instruction

    if is_search_formatting_step_claude:
        final_system_prompt_for_claude = system_instruction.replace("PALEAI_SEARCH_FORMATTING_TASK_MARKER", "").strip()
        if initial_user_prompt:
             final_system_prompt_for_claude += f"\n\n[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"
    else:
        final_system_prompt_for_claude = FRIENDLY_TONE_SYSTEM_PROMPT
        if formatted_memory_info:
            final_system_prompt_for_claude += f"\n\n{formatted_memory_info}"
        if system_instruction:
            final_system_prompt_for_claude += f"\n\n{system_instruction}"
        if initial_user_prompt:
            prefix = "\n\n" if final_system_prompt_for_claude.strip() else ""
            final_system_prompt_for_claude += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」\nこの目的を常に意識して回答してください。"

    api_params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages_for_api,
        "temperature": 0.6
    }
    if final_system_prompt_for_claude.strip():
        api_params["system"] = final_system_prompt_for_claude.strip()

    try:
        print(f"Claude API Request ({model}): System='{(api_params.get('system', 'N/A'))[:100].strip()}', UserMemories: {len(user_memories) if user_memories else 0}, Messages Count={len(messages_for_api)}")
        res = await anthropic_aclient.messages.create(**api_params)

        response_text = ""
        if res.content and isinstance(res.content, list):
            for block in res.content:
                if hasattr(block, 'text'):
                    response_text += block.text

        if not response_text.strip() and hasattr(res, 'stop_reason') and res.stop_reason is not None and res.stop_reason != "end_turn":
             return schemas.IndividualAIResponse(source=f"Claude ({model})", error=f"APIエラーまたは予期しない停止理由。Stop Reason: {res.stop_reason}, Response Content: {res.content}")

        return schemas.IndividualAIResponse(source=f"Claude ({model})", response=response_text if response_text.strip() else "AIからテキスト応答がありませんでした。")
    except Exception as e:
        print(f"Anthropic APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(source=f"Claude ({model})", error=f"API呼び出し中にエラー: {str(e)}")

async def get_cohere_response(
    prompt_text: str,
    preamble: Optional[str] = None,
    model: str = "command-r-plus",
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None
) -> schemas.IndividualAIResponse:
    if not cohere_aclient:
        return schemas.IndividualAIResponse(source=f"Cohere ({model})", error="Cohereクライアントが初期化されていません。")

    final_preamble_for_cohere = FRIENDLY_TONE_SYSTEM_PROMPT
    if preamble:
        final_preamble_for_cohere += f"\n\n{preamble}"
    if initial_user_prompt:
        prefix = "\n\n" if final_preamble_for_cohere.strip() else ""
        final_preamble_for_cohere += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」\nこの目的を常に意識して回答を生成してください。"

    cohere_api_chat_history: List[Dict[str, str]] = []
    if chat_history:
        for msg in chat_history:
            role_map = {"user": "USER", "assistant": "CHATBOT", "ai": "CHATBOT"}
            if msg.get("role") in role_map:
                cohere_api_chat_history.append({"role": role_map[msg["role"]], "message": str(msg.get("content",""))})

    if not prompt_text.strip():
         return schemas.IndividualAIResponse(source=f"Cohere ({model})", error="現在のユーザープロンプトが空です。")

    try:
        print(f"Cohere API Request ({model}): Preamble='{final_preamble_for_cohere[:100].strip() if final_preamble_for_cohere else 'N/A'}...', History Len={len(cohere_api_chat_history)}, Current Message='{prompt_text[:50].strip()}'")
        res = await cohere_aclient.chat(
            message=prompt_text,
            model=model,
            preamble=final_preamble_for_cohere.strip() if final_preamble_for_cohere.strip() else None,
            chat_history=cohere_api_chat_history if cohere_api_chat_history else None,
            temperature=0.7
        )
        return schemas.IndividualAIResponse(source=f"Cohere ({model})", response=res.text)
    except Exception as e:
        print(f"Cohere APIエラー ({model}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(source=f"Cohere ({model})", error=f"API呼び出し中にエラー: {str(e)}")

async def get_perplexity_response(
    prompt_for_perplexity: str, # 呼び出し元で整形済みの完全なプロンプト文字列
    model: str = "sonar-pro", # sonar-pro に固定
) -> schemas.IndividualAIResponse:
    source_name = f"PerplexityAI ({model})"
    if not perplexity_client_sync:
        return schemas.IndividualAIResponse(source=source_name, error="Perplexityクライアントが初期化されていません。")

    if not prompt_for_perplexity.strip():
        return schemas.IndividualAIResponse(source=source_name, error="送信するクエリが空です。")

    try:
        print(
            f"Perplexity AI Request ({model}): Query (first 150 chars)='{prompt_for_perplexity[:150].replace(chr(10), ' ')}...'"
        )

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
                    print(f"Perplexity API ({p_model_name}): 予期しない応答形式 (query): {type(response_data)}, content: {response_data}")
                    return f"Perplexityから予期しない応答形式を受け取りました。"
            except AttributeError as ae:
                print(f"PerplexityClient API呼び出し中にAttributeError (in sync_perplexity_call): {ae}")
                return f"Perplexity APIの呼び出し方法に問題があります (AttributeError): {ae}"
            except Exception as exc_inner:
                print(f"PerplexityClient API呼び出し中に同期関数内でエラー (in sync_perplexity_call): {exc_inner}")
                import traceback
                traceback.print_exc()
                return f"Perplexity APIエラー: {exc_inner}"
        response_text = await run_in_threadpool(
            sync_perplexity_call, perplexity_client_sync, prompt_for_perplexity, model
        )

        if isinstance(response_text, str) and response_text.strip() and \
           not response_text.lower().startswith("perplexity api") and \
           not response_text.lower().startswith("perplexityから予期しない応答形式") and \
           not response_text.lower().startswith("perplexity apiの適切な呼び出しメソッドが見つかりません"):
            import re
            links = re.findall(r"https?://[^\s]+", response_text)
            return schemas.IndividualAIResponse(source=source_name, response=response_text, links=links or None)
        elif isinstance(response_text, str): # エラーメッセージ等が返ってきた場合
             return schemas.IndividualAIResponse(source=source_name, error=response_text)
        else:
            return schemas.IndividualAIResponse(source=source_name, error=f"PerplexityAIから予期しない応答形式 ({type(response_text)}). Text: '{response_text}'")

    except Exception as e:
        print(f"PerplexityAI APIエラー ({model}) (outer try-except): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(source=source_name, error=f"API呼び出し中に予期せぬエラー: {str(e)}")

# --- ここまでが get_perplexity_response 関数の末尾 ---
# main.py の get_gemini_response 関数
async def get_gemini_response(
    prompt_text: str,
    system_instruction: Optional[str] = None,
    model_name: str = 'gemini-1.5-pro-latest',
    chat_history: Optional[List[Dict[str, str]]] = None,
    initial_user_prompt: Optional[str] = None
) -> schemas.IndividualAIResponse:
    source_name = f"Gemini ({model_name})"
    
    # APIキーの存在確認 (環境変数から取得したグローバル変数 GOOGLE_API_KEY を使用)
    if not GOOGLE_API_KEY:
        return schemas.IndividualAIResponse(source=source_name, error="Gemini APIキーが環境変数に設定されていません。")

    active_gemini_model: Optional[genai.GenerativeModel] = None
    try:
        # genai.configure はアプリケーション起動時に一度だけ行うのが理想的。
        # ここでは、configureが既に呼ばれていることを前提とし、モデルのインスタンス化を試みる。
        # もし configure が未実行の場合、genai.GenerativeModel でエラーが発生する可能性がある。
        # その場合は、アプリケーション起動時の初期化処理を見直す。
        active_gemini_model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Geminiモデル '{model_name}' の初期化に失敗: {e}")
        # 起動時に configure されていなかった可能性を考慮し、ここで試みる
        if GOOGLE_API_KEY:
            try:
                print("Gemini: genai.configure() を試行します...")
                genai.configure(api_key=GOOGLE_API_KEY)
                active_gemini_model = genai.GenerativeModel(model_name)
                print("Gemini: モデルの再初期化に成功しました。")
            except Exception as e2:
                print(f"Geminiモデル '{model_name}' の再初期化にも失敗: {e2}")
                import traceback
                traceback.print_exc()
                return schemas.IndividualAIResponse(source=source_name, error=f"Geminiモデルの初期化/設定に失敗: {str(e2)}")
        else: # APIキーがない場合はここでエラー
             return schemas.IndividualAIResponse(source=source_name, error="Gemini APIキーが設定されていません (再試行時)。")


    if not active_gemini_model:
         return schemas.IndividualAIResponse(source=source_name, error="Geminiモデルの取得に重大なエラーが発生しました（初期化後）。")

    contents_for_api: List[Dict[str, Any]] = []
    # ...
    effective_initial_instructions = ""
    is_search_formatting_step_gemini = system_instruction and "PALEAI_SEARCH_FORMATTING_TASK_MARKER" in system_instruction

    if is_search_formatting_step_gemini:
        effective_initial_instructions = system_instruction.replace("PALEAI_SEARCH_FORMATTING_TASK_MARKER", "").strip()
    else:
        effective_initial_instructions = FRIENDLY_TONE_SYSTEM_PROMPT
        if system_instruction:
            effective_initial_instructions += f"\n\n{system_instruction}"

    if initial_user_prompt:
        prefix = "\n\n" if effective_initial_instructions.strip() else ""
        if not is_search_formatting_step_gemini:
            effective_initial_instructions += f"{prefix}[重要] この会話全体の主要な目的は次の通りです: 「{initial_user_prompt}」です。"
        else:
            effective_initial_instructions += f"{prefix}[参考] ユーザーの当初の質問の文脈: 「{initial_user_prompt}」"


    # Gemini APIでは、システム指示はcontentsの最初のuserロールの前に置くか、
    # `GenerativeModel` の `system_instruction` パラメータ (beta) で渡す。
    # ここでは、従来のcontentsに含める方法で、履歴の先頭か、単発プロンプトの先頭に挿入する。

    # contents_for_api の生成ロジック
    if effective_initial_instructions.strip():
        # Geminiはsystemロールを直接サポートしないため、通常は最初のuserメッセージに含めるか、
        # model.start_chat(history=[...], instruction=system_instruction) のように使う。
        # ここでは、履歴の最初のユーザーメッセージにプレフィックスとしてシステム指示を追加するアプローチを試みる。
        # もし履歴が空か、AIから始まる場合は、新しいユーザーメッセージを作成してシステム指示を先頭に置く。

        # system_instruction を model インスタンスに設定する新しい方法 (genaiライブラリのバージョンによる)
        # if hasattr(active_gemini_model, 'system_instruction'):
        #    active_gemini_model.system_instruction = genai.types.Content(parts=[genai.types.Part(text=effective_initial_instructions.strip())])
        # 上記が使えない場合は、従来通りメッセージリストに含める
        pass # 下のロジックで処理


    temp_chat_history = list(chat_history) if chat_history else []

    # 履歴の先頭にシステム指示を組み込む (Geminiの推奨する形式に近づける)
    # systemロールの代わりに、最初のユーザーメッセージの前に置くか、専用パラメータを使用
    # ここでは、履歴のメッセージリストを直接操作して、最初の要素の前にシステム指示的なものをuserロールで追加する形は避ける
    # 代わりに、generate_content_async の contents パラメータを工夫する

    processed_history_for_gemini: List[Dict[str, Any]] = []
    if temp_chat_history:
        for msg in temp_chat_history:
            role = "model" if msg.get("role") in ["ai", "assistant"] else "user"
            processed_history_for_gemini.append({"role": role, "parts": [{"text": str(msg.get("content", ""))}]})

    # 最終的なプロンプトをユーザーロールとして追加
    # システム指示は、`generate_content_async` に渡す `contents` の構成で工夫するか、
    # `active_gemini_model.start_chat(system_instruction=...)` を使う必要がある。
    # `generate_content_async` に直接システムロールを渡すのは非推奨。
    # 代わりに、`system_instruction` パラメータが利用可能ならそれを使うか、
    # 履歴の最初のユーザーメッセージの前にコンテキストとして挿入する。

    # ここでは、contents_for_api を直接構築する
    # 1. system_instruction (もしあれば)
    # 2. chat_history
    # 3. prompt_text (現在のユーザー入力)

    if effective_initial_instructions.strip() and not processed_history_for_gemini:
        # 履歴がなく、システム指示とプロンプトのみの場合
        # Geminiではシステムメッセージをユーザーメッセージの前に置くことが多い
        contents_for_api.append({"role": "user", "parts": [{"text": effective_initial_instructions.strip()}]})
        contents_for_api.append({"role": "model", "parts": [{"text": "承知いたしました。どのようなご用件でしょうか？"}]}) # システム指示に対するモデルの応答例
        contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]})
    elif effective_initial_instructions.strip() and processed_history_for_gemini:
        # 履歴があり、システム指示もある場合
        # 最初のユーザーメッセージの前にシステム指示を置く
        # ただし、Googleのドキュメントでは system_instruction パラメータを推奨
        # ここではcontentsを工夫する例:
        # (注意: この方法はモデルの解釈に依存する可能性がある)
        # contents_for_api.append({"role": "user", "parts": [{"text": effective_initial_instructions.strip() }]}) # システム指示をユーザーメッセージとして
        # contents_for_api.append({"role": "model", "parts": [{"text": "はい、理解しました。"}]}) # それに対するダミーのモデル応答
        # contents_for_api.extend(processed_history_for_gemini)
        # contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]})
        # より推奨されるのは、modelインスタンスのsystem_instructionに設定するか、
        # generate_content_asyncのsystem_instruction引数 (利用可能なら)

        # 今回は、履歴をそのまま使い、最後のユーザープロンプトにシステム指示を付加する（やや苦肉の策）
        contents_for_api.extend(processed_history_for_gemini)
        final_user_prompt_with_instruction = prompt_text
        if not any(item.get("role") == "user" for item in contents_for_api): # 履歴がAIから始まっている場合など
            contents_for_api.insert(0, {"role": "user", "parts": [{"text": effective_initial_instructions.strip() + "\n\n" + prompt_text}]})
        elif contents_for_api: # 履歴の最後のユーザーメッセージに付加、なければ新規追加
            last_message_is_user = contents_for_api[-1]["role"] == "user"
            if last_message_is_user:
                # 最後のユーザーメッセージに結合するのは、そのメッセージの意図を変えてしまう可能性がある
                # そのため、システム指示は別途考慮するか、履歴の先頭に置く設計が望ましい。
                # ここでは、Gemini の場合、system_instruction を使うのがベストプラクティス。
                # それが難しい場合の次善策として、履歴を整形する。
                # 今回は、`active_gemini_model.generate_content_async` に `system_instruction` がないので、
                # `contents` の先頭にユーザーロールでシステム指示を入れ、モデルロールでACKを入れ、その後履歴とプロンプトを入れる形にする。
                contents_for_api = [] # 一旦クリア
                contents_for_api.append({"role": "user", "parts": [{"text": effective_initial_instructions.strip()}]})
                contents_for_api.append({"role": "model", "parts": [{"text": "承知いたしました。指示に従います。"}]}) # AIの受諾応答
                contents_for_api.extend(processed_history_for_gemini) # その後の履歴
                contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]}) # 最後のユーザープロンプト
            else: # 履歴の最後がモデルの場合
                contents_for_api.append({"role": "user", "parts": [{"text": effective_initial_instructions.strip() + "\n\n" + prompt_text}]})
        else: # 履歴がない場合
            contents_for_api.append({"role": "user", "parts": [{"text": effective_initial_instructions.strip() + "\n\n" + prompt_text}]})

    elif processed_history_for_gemini: # システム指示なし、履歴あり
        contents_for_api.extend(processed_history_for_gemini)
        contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]})
    else: # システム指示なし、履歴なし、プロンプトのみ
        contents_for_api.append({"role": "user", "parts": [{"text": prompt_text}]})

    # ユーザーメッセージが必ず存在するように最終チェック
    if not any(item.get("role") == "user" for item in contents_for_api):
        # 通常、ここには到達しないはずだが、フォールバック
        contents_for_api.append({"role": "user", "parts": [{"text": prompt_text or "何か情報を教えてください。"}]})
    # ...
    if not any(item.get("role") == "user" for item in contents_for_api):
        if prompt_text: # chat_history が空でも prompt_text があればそれを最後の砦としてユーザーメッセージとする
             content_to_send_fallback = f"{effective_initial_instructions.strip()}\n\n---\n\n{prompt_text}" if effective_initial_instructions.strip() else prompt_text
             contents_for_api.append({"role": "user", "parts": [{"text": content_to_send_fallback}]})
        else:
            error_detail = f"APIリクエストに有効なユーザーメッセージが含まれていません。Contents: {contents_for_api}, Prompt: {prompt_text}, History: {chat_history is not None}"
            print(f"Geminiデバッグ: {error_detail}")
            return schemas.IndividualAIResponse(source=source_name, error=error_detail)
    
    try:
        print(f"Gemini API Request ({active_gemini_model.model_name}): System/Initial Info (combined)='{effective_initial_instructions[:100].strip() if effective_initial_instructions else 'N/A'}...', Contents Len={len(contents_for_api)}")
        # print(f"Gemini contents_for_api to be sent: {json.dumps(contents_for_api, indent=2, ensure_ascii=False)}")
        
        res = await active_gemini_model.generate_content_async(
            contents=contents_for_api,
            generation_config={"temperature": 0.6}
        )
        
        content_response = ""
        if res.candidates and res.candidates[0].content and res.candidates[0].content.parts:
            content_response = "".join(part.text for part in res.candidates[0].content.parts if hasattr(part, 'text'))
        elif hasattr(res, 'prompt_feedback') and res.prompt_feedback and hasattr(res.prompt_feedback, 'block_reason') and res.prompt_feedback.block_reason: # type: ignore
            block_reason_obj = getattr(res.prompt_feedback, 'block_reason', '不明な理由') # type: ignore
            block_message = getattr(res.prompt_feedback, 'block_reason_message', str(block_reason_obj)) # type: ignore
            error_detail = f"コンテンツ生成がブロックされました: {block_message}"
            print(f"Gemini API: {error_detail}")
            return schemas.IndividualAIResponse(source=source_name, error=error_detail, response=None)
        else:
            print(f"Geminiから有効な応答パーツが見つかりませんでした。Full response: {res}")
            # エラーではなく、単に応答が空、あるいは期待した形式でない場合
            content_response = "" # 空の応答として扱う
            # error メッセージを設定するかどうかは仕様による
            # return schemas.IndividualAIResponse(source=source_name, response="", error="応答にテキストパーツが含まれていません。")

        return schemas.IndividualAIResponse(source=source_name, response=content_response)

    except Exception as e:
        print(f"Gemini APIエラー ({model_name}): {e}")
        import traceback
        traceback.print_exc()
        return schemas.IndividualAIResponse(source=source_name, error=f"API呼び出し中にエラー: {str(e)}")

async def translate_with_deepl(text: str, target_lang: str = "JA") -> str:
    if not deepl_translator:
        raise ValueError("DeepL translator is not configured")
    try:
        result = await run_in_threadpool(deepl_translator.translate_text, text, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"DeepL翻訳エラー: {e}")
        raise




@app.post("/translate", response_model=schemas.TranslationResponse)
async def translate_endpoint(request: schemas.TranslationRequest, current_user: models.User = Depends(get_current_active_user)):
    translated = await translate_with_deepl(request.text, request.target_lang)
    return {"translated_text": translated}

# main.py の末尾近く

# --- main.py にあったチャットメッセージ取得エンドポイントは routers/chat.py に移管したためコメントアウト ---
# @app.get("/chat/sessions/{session_id}/messages", response_model=List[schemas.ChatMessageResponse])
# async def get_chat_session_messages_main( # 関数名変更して区別
#     session_id: int,
#     db: Session = Depends(get_db),
#     current_user: models.User = Depends(get_current_active_user)
# ):
#     # このロジックは routers/chat.py の get_chat_session_messages に移管されています。
#     # chat_session = db.query(models.ChatSession).filter(
#     #     models.ChatSession.id == session_id,
#     #     models.ChatSession.user_id == current_user.id # アクセス制御
#     # ).first()
#     # if not chat_session:
#     #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied")
#     # messages = db.query(models.ChatMessage).filter(
#     #     models.ChatMessage.chat_session_id == session_id
#     # ).order_by(models.ChatMessage.created_at.asc()).all()
#     # return messages
#     pass

@app.post("/collaborative_answer_v2", response_model=schemas.CollaborativeResponseV2)
async def collaborative_answer_mode_endpoint(
    request: schemas.PromptRequestWithHistory, # schemas. を使用
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
) -> schemas.CollaborativeResponseV2: # 戻り値の型ヒントを schemas.CollaborativeResponseV2 に
    original_prompt = request.prompt.strip() # 前後の空白を除去
    mode = request.mode.lower()
    session_id = request.session_id
    desired_char_count = request.char_count

    print(f"\nリクエスト受信: UserID={current_user.id}, SessionID={session_id}, Prompt='{original_prompt[:50].strip()}...', Mode='{mode}'")

    # レスポンスの骨格を先に準備
    response_shell = schemas.CollaborativeResponseV2(
        prompt=original_prompt,
        mode_executed=mode,
        processed_session_id=session_id # session_id が None の場合も初期値として設定 (後で確定値に更新の可能性あり)
    )

    active_session: Optional[models.ChatSession] = None
    initial_user_prompt_for_session: Optional[str] = None

# main.py の /collaborative_answer_v2 内
# ... (既存コード) ...

# main.py の /collaborative_answer_v2 内
# ... (既存コード) ...
    session_id_from_request = request.session_id # 変数名を変更して明確化

    print(f"\nリクエスト受信: UserID={current_user.id}, SessionID(Req)={session_id_from_request}, Prompt='{original_prompt[:50].strip()}...', Mode='{mode}'")

    response_shell = schemas.CollaborativeResponseV2( # レスポンスシェルの初期化を先に
        prompt=original_prompt,
        mode_executed=mode,
        processed_session_id=session_id_from_request # 初期値としてリクエスト時のIDを設定
    )

    active_session: Optional[models.ChatSession] = None
    initial_user_prompt_for_session: Optional[str] = None
    chat_history_for_ai: List[Dict[str, str]] = [] # AIヘルパーに渡す履歴リスト

    # --- 1. チャットセッションの特定または作成 ---
    if session_id_from_request:
        active_session = db.query(models.ChatSession).filter(
            models.ChatSession.id == session_id_from_request,
            models.ChatSession.user_id == current_user.id
        ).first()
        if not active_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="指定されたチャットセッションが見つからないか、アクセス権がありません。")
        print(f"既存チャットセッション使用: ID={active_session.id}, Title='{active_session.title}'")
        response_shell.processed_session_id = active_session.id # レスポンスシェルにも確定IDを設定

        first_user_message_db = db.query(models.ChatMessage).filter( # 変数名を変更
            models.ChatMessage.chat_session_id == active_session.id,
            models.ChatMessage.role == "user"
        ).order_by(models.ChatMessage.created_at.asc()).first()
        if first_user_message_db:
            initial_user_prompt_for_session = first_user_message_db.content
        else: # 既存セッションだが最初のユーザーメッセージがない場合（通常ありえないが）
            initial_user_prompt_for_session = original_prompt # 現在のプロンプトで代用

        # 既存セッションの履歴を取得 (現在のプロンプトはまだ含めない)
        past_messages_db = db.query(models.ChatMessage).filter(
            models.ChatMessage.chat_session_id == active_session.id
        ).order_by(models.ChatMessage.created_at.asc()).all()
        for msg_db in past_messages_db:
            role_for_ai_helper = "assistant" if msg_db.role == "ai" else msg_db.role # AIヘルパー用のロール名に変換
            chat_history_for_ai.append({"role": role_for_ai_helper, "content": msg_db.content})
        print(f"取得した過去メッセージ数 (現在のプロンプトはまだ含まず): {len(chat_history_for_ai)}")

    else: # 新規セッションの場合
        # ★★★ タイトル自動生成 (簡易版) ★★★
        if original_prompt:
            potential_title = original_prompt.splitlines()[0][:50].strip()  # 改行を考慮し、最初の行の先頭50文字
            if not potential_title:  # 空プロンプトや改行のみの場合
                potential_title = "新しいチャット"
        else:
            potential_title = "新しいチャット"  # プロンプトが全くない場合
        # ★★★ ここまで ★★★

        active_session = models.ChatSession(user_id=current_user.id, title=potential_title, status='loading')
        db.add(active_session)
        db.commit()  # 先にセッションをコミットしてIDを確定させる
        db.refresh(active_session)
        print(f"新規チャットセッション作成成功: ID={active_session.id}, Title='{active_session.title}'")
        response_shell.processed_session_id = active_session.id  # レスポンスシェルに確定IDを設定
        initial_user_prompt_for_session = original_prompt  # 新規なので現在のプロンプトが最初のプロンプト

    # --- 2. 過去のチャット履歴の取得 (AI送信用) ---
    #    (このステップは、ユーザーメッセージをDBに保存した後、かつ active_session.id が確定した後に行う方が、
    #     現在のプロンプトを含めずに過去の履歴だけを正確に取得しやすいかもしれません。
    #     しかし、現状のコードフローでは、先に履歴を取得し、後で現在のプロンプトをDB保存時に追加しています。
    #     ここでは、そのフローを維持しつつ、initial_user_prompt_for_session を使うことに主眼を置きます。)
    chat_history_for_ai: List[Dict[str, str]] = []
    if active_session and active_session.id: # 既存セッションでIDが確定している場合のみ
        past_messages_db = db.query(models.ChatMessage).filter(
            models.ChatMessage.chat_session_id == active_session.id
        ).order_by(models.ChatMessage.created_at.asc()).all()
        for msg_db in past_messages_db:
            role_for_ai = "assistant" if msg_db.role == "ai" else msg_db.role
            chat_history_for_ai.append({"role": role_for_ai, "content": msg_db.content})
        print(f"取得した過去メッセージ数 (DB保存前の現在のプロンプトは含まず): {len(chat_history_for_ai)}")

    # --- 3. ユーザーメッセージをDBに保存 ---
    if not active_session:  # 万が一ここでセッションが無い場合に備える
        session_title = (
            original_prompt[:70].strip() + "..." if len(original_prompt) > 70 else original_prompt.strip()
        )
        active_session = models.ChatSession(user_id=current_user.id, title=session_title, status='loading')
        print(
            f"警告: active_sessionが未作成だったため、新規作成します。Title='{active_session.title}'"
        )

    if not active_session.id:
        db.add(active_session)
        db.commit()
        db.refresh(active_session)
        print(f"新規チャットセッションをDBに保存しID確定: ID={active_session.id}")

    # セッションIDが確定しているのでレスポンスにも設定
    response_shell.processed_session_id = active_session.id

    user_message_db = models.ChatMessage(
        chat_session_id=active_session.id,
        user_id=current_user.id,
        role="user",
        content=original_prompt,
    )

    db.add(user_message_db)
    active_session.updated_at = func.now()  # セッションの最終更新日時を更新
    active_session.status = 'loading'
    db.add(active_session)  # 明示的に追加して更新をトラッキング
    db.commit()
    db.refresh(user_message_db)
    db.refresh(active_session)
    print(
        f"ユーザーメッセージ保存成功: MsgID={user_message_db.id}, SessionID={active_session.id}"
    )


    # DB保存後に、AIに渡すチャット履歴を再構築 (現在のプロンプトも含むようにする)
    # ただし、各モードフロー関数側で現在のプロンプトを履歴の最後に追加する方が柔軟性が高い。
    # ここでは、モードフロー関数に渡す chat_history_for_ai は「現在のプロンプトを含まない過去の履歴」のままにする。
    # original_prompt は別途引数で渡す。

    # --- 4. AI連携フローの実行 ---
    try:
        # 各モード実行フロー関数に initial_user_prompt_for_session と chat_history_for_ai を渡す
        # chat_history_for_ai は現在の original_prompt を含まない「それ以前の」履歴
        # original_prompt は現在のユーザー入力として別途渡す
        if mode == "balance":
            response_shell = await run_balance_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai), # 副作用を防ぐためコピーを渡す
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        elif mode in ("search", "search3"):
            response_shell = await run_search_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        elif mode in ("search6", "supersearch"):
            response_shell = await run_super_search_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        elif mode == "code":
            response_shell = await run_code_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        elif mode == "writing":
            response_shell = await run_writing_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        elif mode == "longwriting":
            response_shell = await run_ultra_writing_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session,
                desired_char_count=desired_char_count
            )
        elif mode == "fastchat":
            response_shell = await run_fast_chat_mode_flow(
                original_prompt=original_prompt,
                response_shell=response_shell,
                chat_history_for_ai=list(chat_history_for_ai),
                initial_user_prompt_for_session=initial_user_prompt_for_session
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"無効なモード「{mode}」が指定されました。"
            )

        # --- 5. AIの応答をDBに保存 ---
        # response_shell の中の最終応答フィールド (例: step7_final_answer_v2_openai) を見て判断
        # この部分は、各モードで最終応答が格納されるフィールド名が異なる可能性があるため、
        # モードごとに分岐するか、あるいは共通のフィールドに最終結果を格納するルールにする必要がある。
        # ここでは、balance モードの最終結果を代表として処理する例を示す。
        # 実際には、mode に応じて参照するフィールドを変えるか、
        # 各モード実行関数が final_response のような共通フィールドに結果を格納するようにする。

        # --- 5. AIの応答をDBに保存 ---
        # 各モードで最終応答が格納されるフィールドが response_shell.step7_final_answer_v2_openai に
        # 集約されることを前提とする。
        # 検索モードの場合、step7_final_answer_v2_openai.response が整形済み断片リスト本体、
        # search_summary_text が短いまとめ、search_fragments が生の断片リストとなる。
        # DBには主に step7_final_answer_v2_openai.response を保存する。

        final_ai_response_content_for_db: str = ""
        final_ai_source_text_for_db: str = f"Unknown AI ({mode} mode)"
        final_response_object_for_db: Optional[schemas.IndividualAIResponse] = None

        # 全てのモードで、主要な最終応答は step7_final_answer_v2_openai に格納されることを期待する
        if response_shell.step7_final_answer_v2_openai and response_shell.step7_final_answer_v2_openai.response:
            final_response_object_for_db = response_shell.step7_final_answer_v2_openai
            final_ai_response_content_for_db = final_response_object_for_db.response
            final_ai_source_text_for_db = final_response_object_for_db.source or f"Final Output ({mode.capitalize()} Mode)"
        elif response_shell.overall_error:  # モード実行全体でエラーがあった場合
            final_ai_response_content_for_db = f"処理中にエラーが発生しました: {response_shell.overall_error}"
            final_ai_source_text_for_db = f"Error in {mode.capitalize()} Mode"
        else:  # 有効な応答も全体エラーもない場合
            final_ai_response_content_for_db = "AIから有効な応答がありませんでした。"
            final_ai_source_text_for_db = f"No Valid Response in {mode.capitalize()} Mode"
            # この場合、final_response_object_for_db は None のまま

        # DB保存処理 (この部分は以前の修正から流用・確認)
        if final_ai_response_content_for_db and active_session and active_session.id:
            ai_message_db = models.ChatMessage(
                chat_session_id=active_session.id,
                role="ai",
                content=final_ai_response_content_for_db,  # 整形済み断片リスト本体など
                ai_model=final_ai_source_text_for_db,
                user_id=None
            )
            db.add(ai_message_db)
            active_session.updated_at = func.now()
            active_session.status = 'complete'
            db.add(active_session)  # 明示的なadd
            db.commit()
            db.refresh(ai_message_db)
            db.refresh(active_session)
            print(f"AIレスポンス保存成功: MsgID={ai_message_db.id}, SessionID={active_session.id}, Source='{final_ai_source_text_for_db}'")
        elif response_shell.overall_error:
            print(f"AI処理でエラー発生のためDBへのAI応答保存をスキップ: {response_shell.overall_error}")
            if active_session:  # セッションステータスだけは更新
                active_session.status = 'error'
                db.commit()
                db.refresh(active_session) # refresh を追加
        else:
            print(f"AIからの最終応答が見つからないか内容が空のためDBへのAI応答保存をスキップ: Mode='{mode}'")
            if active_session:
                active_session.status = 'complete_no_response'  # 例えば
                db.commit()
                db.refresh(active_session) # refresh を追加

    except ValueError as ve:
        error_message = f"モード '{mode}' の処理中にエラーが発生しました: {str(ve)}"
        print(f"ValueError in collaborative_answer_mode_endpoint: {error_message}")
        response_shell.overall_error = error_message
        return response_shell # エラー時もレスポンスシェルを返す

    except HTTPException as he: # FastAPIのHTTPExceptionを再raise
        raise he
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_message = f"予期せぬエラーが発生しました: {str(e)}"
        print(f"Unexpected Error in collaborative_answer_mode_endpoint: {error_message}\nTrace: {error_trace}")
        # response_shell.overall_error = f"サーバー内部で予期せぬエラーが発生しました。" # ユーザーにはシンプルなエラーを
        # エラー内容をもう少し具体的にクライアントに返したい場合は調整
        response_shell.overall_error = f"サーバー処理中にエラーが発生しました: {str(e)}"

        # 開発中は詳細なエラーを返すことも検討（本番ではセキュリティ上非推奨）
        # response_shell.overall_error = error_message
        return response_shell

    print(f"Endpoint (normal path) is about to return response_shell.")
    if response_shell:
        print(f"Final content of response_shell: {response_shell.model_dump_json(indent=2)}")
    else:
        # このパスには到達しないはずだが、万が一 response_shell が None になった場合のログ
        print("Error: response_shell is None before returning from endpoint.")
        # 何らかのデフォルトエラーレスポンスを返すか、HTTPExceptionを発生させるべき
        raise HTTPException(status_code=500, detail="サーバー内部エラー: レスポンスオブジェクトがnullです。")

    return response_shell


# --- 6段階超検索モード ---
async def run_super_search_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    max_refinement_loops: int = 1 # 初期検索 + 不足観点からの再検索1回 = 計2ループ相当
) -> schemas.CollaborativeResponseV2:
    print("\n--- 新・超検索特化モード（ペイルの叡智）開始 ---")
    response_shell.search_summary_text = None
    response_shell.search_mode_warnings = {}
    response_shell.step7_final_answer_v2_openai = None

    current_chat_history_with_prompt = list(chat_history_for_ai) # コピー
    current_chat_history_with_prompt.append({"role": "user", "content": original_prompt})

    all_collected_fragments: List[schemas.IndividualAIResponse] = []
    queries_to_process: List[str] = [original_prompt]
    processed_queries: set[str] = set() # 無限ループ防止

    try:
        for loop_count in range(max_refinement_loops + 1): # 初期検索 + 再検索ループ
            current_round_queries = [q for q in queries_to_process if q not in processed_queries]
            if not current_round_queries:
                print(f"超検索モード ループ {loop_count + 1}: 新規クエリがないため終了。")
                break

            queries_to_process = [] # 次のラウンドのためにクリア
            print(f"超検索モード ループ {loop_count + 1}: 処理対象クエリ数 = {len(current_round_queries)}")

            search_tasks = []
            for query_to_search in current_round_queries:
                processed_queries.add(query_to_search)
                print(f"  並列検索タスク準備中 (クエリ: {query_to_search[:50]}...)")

                # Perplexity検索タスク (より詳細な情報を求めるプロンプト)
                perplexity_prompt = (
                    f"ユーザーの主要な関心事は「{original_prompt}」です。"
                    f"現在調査中の具体的な観点は「{query_to_search}」です。"
                    f"この観点について、ウェブ、ニュース、SNS投稿、レビュー、学術論文の抄録など、考えうるあらゆる情報源から、「可能な限り原文のまま、一切省略せずに」詳細な情報を収集してください。"
                    f"各情報には、出典URL、発行日、著者/発信者名、コンテンツの種類（例：ニュース記事、SNS投稿、レビュー原文など）を必ず付記してください。"
                    f"情報の網羅性と詳細度を最優先し、多様な視点からの情報を集めてください。"
                )
                search_tasks.append(
                    get_perplexity_response(prompt_for_perplexity=perplexity_prompt, model="sonar-reasoning-pro") # 高性能モデル
                )

                # (オプション) Gemini Web Search など他の検索APIタスクも追加
                # gemini_web_search_prompt = f"「{query_to_search}」に関する最新かつ多様な情報をウェブから収集し、それぞれの情報源（URL、サイト名、日付など）と共に詳細をリストアップしてください。要約は不要です。"
                # search_tasks.append(get_gemini_response(prompt_text=gemini_web_search_prompt, system_instruction="PALEAI_SEARCH_FORMATTING_TASK_MARKER\nWeb情報収集専門AI", model_name="gemini-pro")) # Geminiの検索に適したモデル

            if not search_tasks:
                if loop_count == 0 and not all_collected_fragments:
                    response_shell.overall_error = "超検索モード：初期クエリで検索タスクを生成できませんでした。"
                    return response_shell
                else:
                    print("超検索モード：これ以上処理する新規クエリがありません。")
                    break

            print(f"  {len(search_tasks)}件の並列検索タスクを実行します...")
            search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)
            print(f"  並列検索タスク完了。結果数: {len(search_results_raw)}")

            # 並列検索結果を整理して all_collected_fragments に追加
            query_idx_for_tasks = 0 # search_tasksとsearch_results_rawの対応を取るため
            for raw_res in search_results_raw:
                # current_round_queriesのどのクエリに対応する結果かを特定する必要がある
                # 簡単のため、タスク作成順と結果の順が一致すると仮定
                original_search_query = current_round_queries[query_idx_for_tasks] if query_idx_for_tasks < len(current_round_queries) else "不明なクエリ"
                query_idx_for_tasks += 1

                frag_to_add = None
                if isinstance(raw_res, Exception):
                    print(f"  並列検索中にエラー (クエリ: {original_search_query}): {raw_res}")
                    frag_to_add = schemas.IndividualAIResponse(
                        source=f"並列検索エラー ({original_search_query[:20]}...)", query=original_search_query, error=str(raw_res)
                    )
                elif isinstance(raw_res, schemas.IndividualAIResponse):
                    # ここでPerplexity等のAPI応答からメタデータ（URL、日付、著者など）を抽出・設定する
                    # この例では簡略化し、応答テキストとリンクのみを扱う
                    frag_to_add = schemas.IndividualAIResponse(
                        source=raw_res.source or "不明な検索ソース",
                        response=raw_res.response,
                        links=raw_res.links,
                        error=raw_res.error,
                        query=original_search_query,
                        intent=f"超検索 ループ{loop_count+1}",
                        source_url=raw_res.links[0] if raw_res.links else None, # 代表URL (仮)
                        # published_date, author, content_type はAPI応答のパースが必要
                    )
                else:
                    frag_to_add = schemas.IndividualAIResponse(
                        source="不明な検索結果型", query=original_search_query, error=f"予期しない型: {type(raw_res)}"
                    )

                if frag_to_add:
                    all_collected_fragments.append(frag_to_add)

            # --- ステップ2 (ループ内): 不足論点/矛盾/信頼性等を分析し、次のクエリ候補を生成 ---
            if loop_count < max_refinement_loops: # 最終ループでは不足観点抽出は不要
                print(f"\n超検索モード ループ {loop_count + 1} - 分析と次クエリ生成中...")
                current_all_info_text = ""
                for idx, frag in enumerate(all_collected_fragments):
                    if frag.response:
                        frag_header = f"--- 情報断片 {idx+1} (ソース: {frag.source}, クエリ: {frag.query or 'N/A'}) ---\n"
                        current_all_info_text += f"{frag_header}{frag.response}\n"
                        if frag.source_url: current_all_info_text += f"  URL: {frag.source_url}\n"
                        if frag.published_date: current_all_info_text += f"  日付: {frag.published_date}\n"
                        current_all_info_text += "\n"

                if not current_all_info_text.strip():
                    print("  収集情報が空のため、分析と次クエリ生成をスキップ。")
                else:
                    analysis_system_prompt = ( # キャラ口調なし
                        "PALEAI_SEARCH_FORMATTING_TASK_MARKER\n"
                        "あなたは高度な情報分析AIです。複数の情報源から得られた複雑な情報を比較検討し、その情報の完全性、一貫性、信頼性、客観性を評価する専門家です。\n"
                        "あなたの主なタスクは、以下の通りです。\n"
                        "1.  **不足している論点や情報**: まだカバーされていない重要な側面や、さらに深掘りすべき具体的なトピックを特定し、それらを新しい検索クエリとして使える形で【3つ以内、各20字程度で】提案してください。\n"
                        "2.  **情報間の矛盾や食い違い**: 異なる情報源間で明確に矛盾している点があれば、具体的に指摘してください。（出力に含める）\n"
                        "3.  **信頼性に疑問がある情報**: 情報源の信憑性が低い、データが古い、客観性に欠けるなど、信頼性に懸念がある情報があれば指摘してください。（出力に含める）\n"
                        "4.  **潜在的なバイアス**: 特定の意見に偏っている、一方的な視点のみが提示されているなど、バイアスが疑われる箇所があれば指摘してください。（出力に含める）\n"
                        "5.  **要再検証・最新情報確認が必要な観点**: 時間経過により情報が変化している可能性が高いトピックや、最新の動向を確認すべき点を挙げてください。（出力に含める）\n"
                        "出力は、まず「新たな検索クエリ候補：」として1で提案されたクエリリストを提示し、その後「分析結果と指摘事項：」として2～5の分析結果を記述してください。"
                    )
                    analysis_user_prompt = (
                        f"ユーザーの主要な関心事は「{original_prompt}」です。\n"
                        f"以下は、これまでに収集された関連情報群です。\n\n"
                        f"--- 収集情報全体 ---\n{current_all_info_text.strip()}\n--- 収集情報全体ここまで ---\n\n"
                        f"上記のシステム指示に従い、この全情報を徹底的に分析し、新たな検索クエリ候補と、その他の分析結果・指摘事項を提示してください。"
                    )
                    # Claude Opus または GPT-4 Turbo/GPT-4o が適任
                    analysis_res = await get_claude_response(
                        prompt_text=analysis_user_prompt,
                        system_instruction=analysis_system_prompt,
                        model="claude-3-opus-20240229",
                        initial_user_prompt=initial_user_prompt_for_session
                    )

                    if analysis_res.response:
                        # 分析結果全体を一時的な断片として保存
                        all_collected_fragments.append(schemas.IndividualAIResponse(
                            source="Claude/GPT 情報分析・評価", response=analysis_res.response, intent="情報分析と次クエリ候補"
                        ))
                        # 分析結果から新しいクエリ候補を抽出 (AIの出力形式に依存)
                        # 例: "新たな検索クエリ候補：" の後に続く行から抽出
                        # ここでは簡易的に、応答全体からキーワードらしきものを抽出する。実際のパースはAIの出力形式に合わせること。
                        raw_queries_section = ""
                        if "新たな検索クエリ候補：" in analysis_res.response:
                            raw_queries_section = analysis_res.response.split("新たな検索クエリ候補：", 1)[1].split("分析結果と指摘事項：", 1)[0]

                        for line in raw_queries_section.splitlines():
                            clean_query = line.strip().lstrip("-・* ").rstrip("?")
                            if clean_query and len(clean_query) > 3 and len(clean_query) < 50 and clean_query not in processed_queries:
                                if len(queries_to_process) < 3: # 次のラウンドのクエリ数を制限
                                    queries_to_process.append(clean_query)
                        print(f"  次のループで処理する新規クエリ候補: {queries_to_process}")
                    if analysis_res.error:
                        print(f"  情報分析・評価ステップでエラー: {analysis_res.error}")
                        all_collected_fragments.append(schemas.IndividualAIResponse(
                            source="Claude/GPT 情報分析・評価エラー", error=analysis_res.error
                        ))

            if not queries_to_process and loop_count < max_refinement_loops : # 次のクエリがないが、まだループが残っている場合
                print("次の検索クエリ候補が見つかりませんでした。ループを早期終了します。")
                break


        # Webスクレイピングや論文API、SNS APIの自動活用 (このフェーズは高度な実装が必要なため、今回は概念のみ)
        # if loop_count > 0 : # 例えば2ループ目以降で検討
        #    print("  (概念) Webスクレイピング、論文API、SNS APIの活用を検討...")
        #    # 特定のURLが見つかればスクレイピング、キーワードで論文検索など

        # --- ループ終了後: 全取得情報を最終的に整形・分類 ---
        print("\n超検索特化モード: 全情報の最終整形・分類中...")
        if not any(f.response for f in all_collected_fragments):
            response_shell.overall_error = "超検索モードで情報が収集できませんでした。"
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="SuperSearch Orchestrator", error="情報収集ステップで有効な結果が得られませんでした。"
            )
            response_shell.search_fragments = all_collected_fragments
            return response_shell

        final_all_fragments_text_for_formatting = ""
        for idx, frag in enumerate(all_collected_fragments):
            # if frag.response: # 有効な応答のみ整形対象とする (分析結果なども含む)
            frag_header = (
                f"--- 情報断片 {idx+1}: (ソース: {frag.source}) ---\n"
                f"  元クエリ: {frag.query or 'N/A'}\n"
                f"  意図/種類: {frag.intent or 'N/A'}\n"
            )
            frag_content = frag.response or frag.error or "内容なし"
            frag_metadata_lines = []
            if frag.source_url: frag_metadata_lines.append(f"  - 出典URL: {frag.source_url}")
            if frag.published_date: frag_metadata_lines.append(f"  - 発行日: {frag.published_date}")
            if frag.author: frag_metadata_lines.append(f"  - 著者/発信者: {frag.author}")
            if frag.content_type: frag_metadata_lines.append(f"  - コンテンツ種類: {frag.content_type}")
            # 信頼性・バイアスラベル (もしあれば)
            if frag.reliability_label: frag_metadata_lines.append(f"  - 信頼性評価(仮): {frag.reliability_label}")
            if frag.bias_label: frag_metadata_lines.append(f"  - バイアス評価(仮): {frag.bias_label}")
            if frag.issues_detected: frag_metadata_lines.append(f"  - 指摘事項: {', '.join(frag.issues_detected)}")

            frag_links_text = "\n".join([f"  - 関連リンク: {link}" for link in frag.links]) if frag.links and not frag.source_url else ""

            final_all_fragments_text_for_formatting += f"{frag_header}{frag_content}\n"
            if frag_metadata_lines: final_all_fragments_text_for_formatting += "\n".join(frag_metadata_lines) + "\n"
            if frag_links_text: final_all_fragments_text_for_formatting += frag_links_text + "\n"
            final_all_fragments_text_for_formatting += "\n\n"

        if not final_all_fragments_text_for_formatting.strip():
            response_shell.overall_error = "超検索モード：最終整形対象となる有効な情報がありませんでした。"
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(source="SuperSearch Orchestrator", error="整形対象情報なし")
            response_shell.search_fragments = all_collected_fragments
            return response_shell

        final_formatting_system_prompt = (
            "PALEAI_SEARCH_FORMATTING_TASK_MARKER\n"
            "あなたは、提供された膨大な量の情報断片とAIによる分析結果を、ユーザーが詳細に理解できるよう整理・構造化する、最高位のAIリサーチアナリスト兼編集長です。\n"
            "あなたの使命は、以下の指示に厳密に従って、包括的で詳細な情報レポートを作成することです。\n"
            "1. **情報の完全なインライン展開と構造化**: 提供された全ての情報断片（原文抜粋、URL、日付、発信者、SNS投稿、レビュー本文、AIによる分析コメントなど）について、その**内容全文を一切省略・圧縮・要約することなく、そのままの形で**レポートに含めてください。「内容: (上記に全文掲載済み)」のような参照形式は使用せず、全てのテキストをインラインで展開してください。\n"
            "2. **多次元的な分類とメタデータ**: 各情報断片を、以下の観点に基づいて多次元的に分類し、それぞれの情報を明記してください。\n"
            "    - **情報源**: (例: PerplexityAI 初期検索結果, Claude 情報分析・評価, PerplexityAI 追加検索結果(クエリ:XXX) など、入力で提供されたソース名)\n"
            "    - **元の検索クエリ**: (もしあれば)\n"
            "    - **収集意図/種類**: (もしあれば)\n"
            "    - **出典URL**: (もしあれば、必ず記載)\n"
            "    - **発行日/投稿日**: (もしあれば、必ず記載)\n"
            "    - **著者/発信者**: (もしあれば、必ず記載)\n"
            "    - **コンテンツの種類**: (例: ニュース記事、学術論文抄録、SNS投稿、製品レビュー、ウェブページ、AI分析コメントなど)\n"
            "    - **信頼性・バイアス等 (AI分析結果より)**: (もしあれば、例: 「信頼性: 高」「バイアス: 肯定的意見に偏り」「指摘事項: 要追加検証」など、AI分析ステップで付与されたラベルやコメントをそのまま引用または要約して記載)\n"
            "    - **主要トピック/サブトピック**: (内容に基づいて適切にグルーピング)\n"
            "    - **時系列**: (可能な範囲で、発行日やイベント発生順など)\n"
            "    - **意見の傾向**: (意見が含まれる情報の場合、肯定的/否定的/中立的など)\n"
            "   これらの情報を各断片に付随させ、全体を論理的かつ視覚的に分かりやすい構造（例: 大見出し、中見出し、箇条書き、表などを適切に活用）で提示してください。\n"
            "3. **解釈の排除**: あなた自身の解釈、意見、追加情報は一切含めず、客観的な情報の整理と構造化された提示に徹してください。\n"
            "4. **最終サマリーと留意点の生成**: 全ての構造化された情報断片を提示した後、**必ず最後に**「【最終サマリーと留意点】」という見出しを付け、その後に2～3段落程度の総合的な概要（収集された情報の主要なポイント、多様な視点、特筆すべき傾向など）と、情報全体の信頼性やバイアスに関する一般的な注意点（AI分析結果や収集情報全体から判断される留意事項など）を記述してください。この部分以外では、絶対に情報を要約したり、個人的な意見を述べたりしないでください。\n"
            "出力は、まず構造化・分類された情報断片の完全なリスト（内容全文とメタデータを含む）、その次に「【最終サマリーと留意点】」とそれに続く記述、という形式にしてください。"
        )
        final_formatting_user_prompt = (
            f"ユーザーの元の主要な関心事は「{original_prompt}」です。\n"
            f"以下に、この関心事について多角的に収集・分析された膨大な量の情報断片群があります。\n\n"
            f"--- 全情報断片・分析結果ここから ---\n{final_all_fragments_text_for_formatting.strip()}\n--- 全情報断片・分析結果ここまで ---\n\n"
            f"上記のシステム指示に厳密に従い、これらの全情報を一切省略・要約せず、多次元的に分類・整形し、提示してください。"
            f"特に、各情報源のURL、発行日、発信者といったメタ情報は可能な限り保持し、表示してください。"
            f"最後に2～3段落の「最終サマリーと留意点」だけを「【最終サマリーと留意点】」という見出しで付与してください。"
        )

        # GPT-4o や Claude Opus, Gemini 1.5 Pro など高性能モデル推奨
        final_formatting_res = await get_gemini_response(
            prompt_text=final_formatting_user_prompt,
            system_instruction=final_formatting_system_prompt,
            model_name="gemini-1.5-pro-latest",
            initial_user_prompt=initial_user_prompt_for_session
        )

        if final_formatting_res.response:
            summary_marker_super = "【最終サマリーと留意点】"
            formatted_super_fragments_display = final_formatting_res.response
            super_summary_text = "（AIによる自動最終サマリーと留意点はありませんでした）" # デフォルト
    
            # 信頼性・バイアス警告の抽出 (簡易的な例、AIの出力形式に依存)
            extracted_warnings = {}
            # 例えば、整形済み応答から特定のキーワードで始まるセクションを探すなど
            # if "信頼性に関する警告：" in formatted_super_fragments_display:
            #     extracted_warnings["reliability"] = formatted_super_fragments_display.split("信頼性に関する警告：",1)[1].split("\n\n",1)[0]
            # if "バイアスに関する指摘：" in formatted_super_fragments_display:
            #     extracted_warnings["bias"] = formatted_super_fragments_display.split("バイアスに関する指摘：",1)[1].split("\n\n",1)[0]
            # response_shell.search_mode_warnings = extracted_warnings
    
            if summary_marker_super in formatted_super_fragments_display:
                parts = formatted_super_fragments_display.split(summary_marker_super, 1)
                formatted_super_fragments_display = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    super_summary_text = parts[1].strip()
                else:
                    super_summary_text = "最終サマリーと留意点部分が空でした。"
            else:
                print(f"警告: 超検索の最終整形結果に{summary_marker_super}マーカーが見つかりませんでした。")
    
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="超検索 最終整形結果 (Gemini/GPT-4o/Claude Opus)",
                response=formatted_super_fragments_display, # 分類・整形された断片情報リスト本体
                query=original_prompt,
                intent="超検索 全情報整形済みリスト"
            )
            response_shell.search_summary_text = super_summary_text # 分離したまとめと留意点
        else:
                response_shell.overall_error = "超検索モードの最終的な情報の整形・分類に失敗しました。"
                response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                    source="SuperSearch Orchestrator", error=f"超検索最終整形ステップでエラー: {final_formatting_res.error or '応答なし'}"
                )
    
                response_shell.search_fragments = all_collected_fragments  # 生の断片情報もレスポンスに含める
                print("--- 新・超検索特化モード（ペイルの叡智）終了 ---")

    except ValueError as ve:
        error_message = f"超検索特化モードの処理中にエラー: {str(ve)}"
        print(error_message)
        response_shell.overall_error = error_message
        response_shell.search_fragments = all_collected_fragments
        if not response_shell.step7_final_answer_v2_openai:
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="SuperSearch Error", error=str(ve)
            )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_message = f"超検索特化モードで予期せぬエラー: {str(e)}"
        print(f"{error_message}\nTrace: {error_trace}")
        response_shell.overall_error = "サーバー内部で予期せぬエラーが発生しました（超検索）。"
        response_shell.search_fragments = all_collected_fragments
        if not response_shell.step7_final_answer_v2_openai:
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="SuperSearch Unexpected Error", error=str(e)
            )

        return response_shell

    return response_shell


# --- 各モード実行フロー関数 ---
# ↓↓↓ 変更点: 各モード実行フロー関数に chat_history_for_ai 引数を追加 ↓↓↓
# main.py の run_balance_mode_flow 関数の修正
async def run_balance_mode_flow(
    original_prompt: str,                            # 現在のユーザーの具体的な質問
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],     # これまでの会話履歴 (現在のプロンプトは含まない)
    initial_user_prompt_for_session: Optional[str]   # このチャットセッション全体の最初のユーザーリクエスト
) -> schemas.CollaborativeResponseV2:

    print("\n--- バランスモード開始 ---")

    # このターンでAIに渡すための完全なチャット履歴を作成 (過去の履歴 + 現在のプロンプト)
    current_chat_history_for_this_turn = list(chat_history_for_ai) # コピーを作成
    current_chat_history_for_this_turn.append({"role": "user", "content": original_prompt})
    print(f"Balance Mode: このターンでAIに渡す完全な履歴は {len(current_chat_history_for_this_turn)} 件")


    # 各ステップの結果を格納する変数の初期化 (変更なし)
    step1_res = schemas.IndividualAIResponse(source="OpenAI (Initial Draft)", error="未実行")
    step2_res = schemas.IndividualAIResponse(source="Claude (Review 1)", error="未実行")
    step3_res = schemas.IndividualAIResponse(source="Cohere (Improved Draft)", error="未実行")
    step4_res = schemas.IndividualAIResponse(source="PerplexityAI (Fact Check/Research)", error="未実行")
    step5_res = schemas.IndividualAIResponse(source="Gemini (Synthesized Answer 1)", error="未実行")
    step6_res = schemas.IndividualAIResponse(source="Claude (Review 2)", error="未実行")
    step7_res = schemas.IndividualAIResponse(source="OpenAI (Final Polished Answer)", error="未実行")

    try:
        # ステップ1: OpenAI - 初期回答の草案生成
        print("ステップ1: OpenAIによる初期回答生成中...")
        step1_system_prompt = (
            "あなたは、ユーザーからの質問とこれまでの会話の流れ、そして会話全体の主要な目的を深く理解し、"
            "まずは網羅的で客観的な情報に基づいた基本的な回答の草案を、序論・本論（複数の主要点）・結論の形式で構造化して作成するAIアシスタントです。"
        )
        step1_res = await get_openai_response(
            prompt_text=original_prompt, # 現在のプロンプト (get_openai_response内で履歴の最後に追加される想定)
            system_role_description=step1_system_prompt,
            model="gpt-4o", # モデル指定
            chat_history=list(current_chat_history_for_this_turn), # このターン用の完全な履歴
            initial_user_prompt=initial_user_prompt_for_session  # セッション最初の質問
        )
        response_shell.step1_initial_draft_openai = step1_res
        if step1_res.error or not step1_res.response:
            raise ValueError(f"Balance Mode - Step1 (OpenAI初期回答) 失敗: {step1_res.error or '応答なし'}")
        print(f"OpenAI 回答草案 (冒頭): {step1_res.response[:100].strip()}...")

        # ステップ2: Claude - 初期回答の批判的レビュー
        print("\nステップ2: Claudeによる批判的レビュー中...")
        step2_system_instruction_claude = (
            "あなたはAIの回答を非常に厳しく、建設的に批判・評価する専門のレビュアーです。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であることを念頭に置いてください。\n"
            "提示された「AIの初期回答草案」を、以下の観点からレビューし、具体的な改善提案を【番号付きリスト形式で5つ以上】挙げてください。\n"
            "- 元のユーザーの質問および会話全体の目的に対して、回答が適切かつ十分か。\n"
            "- 論理の飛躍、矛盾点、根拠の薄弱さ、情報の偏りはないか。\n"
            "- 表現の曖昧さ、分かりにくい部分はないか。\n"
            "- さらに情報を深掘りすべき点、追加すべき視点はないか。\n"
            "あなたのレビューは、次のAIがこの草案を大幅に改善するための重要な土台となります。"
        )
        step2_prompt_for_claude = f"""ユーザーの現在の質問: 「{original_prompt}」
これまでの会話履歴:
{json.dumps(chat_history_for_ai, ensure_ascii=False, indent=2)}

上記を踏まえたAIの初期回答草案 (OpenAI作成):
「{step1_res.response}」

この初期回答草案を、あなたが持つ専門的なレビュアーとしての観点から、上記のシステム指示に従って詳細にレビューし、具体的な改善提案をしてください。
"""
        step2_res = await get_claude_response(
            prompt_text=step2_prompt_for_claude,
            system_instruction=step2_system_instruction_claude,
            model="claude-3-opus-20240229",
            # Claudeの場合、chat_history を渡すか、プロンプトに含めるかは設計による。
            # ここでは主要な情報はプロンプトに含めたので、chat_historyは渡さないか、あるいは
            # current_chat_history_for_this_turn を渡してClaude側で解釈させる。
            # 今回は、レビュー対象のテキストが主なので、履歴は必須ではないかもしれない。
            initial_user_prompt=initial_user_prompt_for_session # system_instructionに含めたが念のため
        )
        response_shell.step2_review_claude = step2_res
        if step2_res.error or not step2_res.response:
            raise ValueError(f"Balance Mode - Step2 (Claudeレビュー1) 失敗: {step2_res.error or '応答なし'}")
        print(f"Claude レビュー1 (冒頭): {step2_res.response[:100].strip()}...")

        # ステップ3: Cohere - 第1改善案の作成
        print("\nステップ3: Cohereによる第1改善案作成中...")
        step3_preamble_cohere = (
            "あなたは、提示された「AIの初期回答草案」と、それに対する「詳細な批判的レビュー」を深く理解し、"
            "レビューでの指摘事項を具体的に反映させて、草案を大幅に改善・再構築する編集専門のAIです。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、"
            f"ユーザーの現在の質問は「{original_prompt}」であることを常に念頭に置いてください。\n"
            "文章全体の論理性、情報の正確性、具体性、網羅性、そして表現の明瞭さを向上させることを目指してください。"
        )
        step3_prompt_for_cohere = f"""以下の情報があります。
1.  ユーザーの現在の質問: 「{original_prompt}」
2.  これまでの会話履歴:
{json.dumps(chat_history_for_ai, ensure_ascii=False, indent=2)}
3.  AIによる初期回答草案 (OpenAI作成):
「{step1_res.response}」
4.  上記草案への詳細な批判的レビューと改善提案 (Claude作成):
「{step2_res.response}」

これらの情報をすべて考慮し、Claudeのレビューで指摘された改善点を具体的に反映させ、OpenAIの初期回答草案を全面的に修正・拡張し、より質の高い「第1改善案」を作成してください。
"""
        step3_res = await get_cohere_response(
            prompt_text=step3_prompt_for_cohere, # Cohereのmessageパラメータに相当
            preamble=step3_preamble_cohere,
            model="command-r-plus",
            # Cohere の chat_history は USER/CHATBOT の交互形式。
            # ここでは current_chat_history_for_this_turn を Cohere 形式に変換して渡すか、
            # 主要なコンテキストはプロンプトに含めたので、履歴は渡さない選択も。
            # バランスモードでは、各AIが前のAIの出力を参照するため、全履歴を毎回渡す必要性は低いかもしれない。
            # initial_user_prompt は preamble に含めた。
            chat_history=list(current_chat_history_for_this_turn) # 渡す場合は整形が必要
        )
        response_shell.step3_improved_draft_cohere = step3_res
        if step3_res.error or not step3_res.response:
            raise ValueError(f"Balance Mode - Step3 (Cohere改善案) 失敗: {step3_res.error or '応答なし'}")
        print(f"Cohere 第1改善案 (冒頭): {step3_res.response[:100].strip()}...")

        # ステップ4: Perplexity AI - 事実確認と追加情報収集
        print("\nステップ4: Perplexity AIによる情報収集と提示中...")
        step4_prompt_for_perplexity = (
            f"ユーザーの最初の質問は「{initial_user_prompt_for_session}」で、現在の質問は「{original_prompt}」です。\n"
            f"これまでのAIによる議論（特にCohereが作成した改善案とClaudeによるレビュー）を踏まえ、さらに回答の質を高めるために、以下の点について最新かつ信頼性の高い情報をウェブから収集し、簡潔にまとめてください。\n"
            f"Claudeのレビューで指摘された情報不足の点: 「{step2_res.response[:200].strip()}...」 (レビューの関連部分)\n"
            f"Cohereの改善案: 「{step3_res.response[:200].strip()}...」 (改善案の関連部分)\n"
            f"これらの内容を補強し、事実誤認がないか確認するために必要な情報を調査し、結果と情報源のURL（もしあれば）を提示してください。"
        )
        step4_res = await get_perplexity_response(
            prompt_for_perplexity=step4_prompt_for_perplexity,
            model="sonar-reasoning-pro", # 推奨モデル
            # Perplexity は履歴をプロンプトに含める形式なので、chat_historyとinitial_user_promptを直接渡すのではなく、
            # 上記のようにプロンプト文字列に情報を組み込む。
        )
        response_shell.step4_comprehensive_answer_perplexity = step4_res
        if step4_res.error or (step4_res.response is not None and len(step4_res.response.strip()) < 10 and not step4_res.error): # 応答が極端に短い場合も考慮
            print(f"Perplexity AIからの応答が不十分またはエラー: {step4_res.error or '応答が短すぎます/ありません'}")
            # エラーがあっても処理を続行し、エラー情報を記録する
            step4_res.response = step4_res.response or "" # Noneなら空文字に
        elif step4_res.response:
            print(f"Perplexity AI リサーチ結果 (冒頭): {step4_res.response[:100].strip()}...")
        else: # error もなく response も None (または空) の場合
            step4_res.response = "" # Noneなら空文字に
            print("Perplexity AIから応答がありませんでした（エラーもなし）。")


        # ステップ5: Gemini - 「第1最終回答」の統合・編集
        print("\nステップ5: Geminiによる「第1最終回答」生成中...")
        step5_system_instruction_gemini = (
            "あなたは、ユーザーからの質問に対し、複数のAIによる多角的な検討とリサーチを経て、質の高い回答を構築するAI編集長です。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの現在の質問は「{original_prompt}」です。\n"
            "提供された全ての情報（初期草案、レビュー、改善案、追加リサーチ情報）を慎重に吟味し、矛盾を解消し、読者にとって最も価値のある「第1最終回答」を構築してください。\n"
            "批判的思考と高度な編集能力を発揮し、論理的で一貫性のある、正確かつ詳細で網羅的な、分かりやすい文章にまとめてください。"
        )
        step5_prompt_for_gemini = f"""ユーザーの現在の質問: 「{original_prompt}」
これまでの会話履歴の要点:
{json.dumps(chat_history_for_ai[-3:], ensure_ascii=False, indent=2)} # 直近3件の履歴例

提供情報:
1.  AIによる初期回答草案 (OpenAI作成): 「{step1_res.response}」
2.  上記草案への批判的レビュー1 (Claude作成): 「{step2_res.response}」
3.  レビューを反映した改善案 (Cohere作成): 「{step3_res.response}」
4.  追加リサーチ情報 (Perplexity AI作成): 「{step4_res.response if step4_res.response else "Perplexityからの追加情報はありませんでした。"}」

これらの情報をすべて活用し、元のユーザープロンプトに対する「第1最終回答」を、上記のシステム指示に従って生成してください。
"""
        step5_res = await get_gemini_response(
            prompt_text=step5_prompt_for_gemini,
            system_instruction=step5_system_instruction_gemini,
            model_name="gemini-1.5-pro-latest", # 推奨モデル
            chat_history=list(current_chat_history_for_this_turn), # ここでは完全な履歴を渡すことも検討
            initial_user_prompt=initial_user_prompt_for_session
        )
        response_shell.step5_final_answer_gemini = step5_res
        if step5_res.error or not step5_res.response:
            raise ValueError(f"Balance Mode - Step5 (Gemini第1最終回答) 失敗: {step5_res.error or '応答なし'}")
        print(f"Gemini 第1最終回答 (冒頭): {step5_res.response[:100].strip()}...")

        # ステップ6: Claude - 「第1最終回答」の再レビュー
        print("\nステップ6: Claudeによる「第1最終回答」の再レビュー中...")
        step6_system_instruction_claude_review2 = (
            "あなたはAIが生成した高度な回答をさらに磨き上げるための最終レビューを行う、超一流の編集・校閲AIです。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの現在の質問は「{original_prompt}」であることを忘れないでください。\n"
            "提示された「第1最終回答」を、さらに高いレベルの品質に引き上げるために、非常に厳しく、多角的な視点から再度レビューしてください。\n"
            "特に以下の点に注目し、具体的な改善点を【番号付きリスト形式で3つ以上】挙げてください。\n"
            "- 論理の一貫性と飛躍のなさ、情報の正確性\n"
            "- 情報の深さと洞察力、新規性\n"
            "- 表現の洗練度、明瞭さ、説得力\n"
            "- 読者への訴求力と分かりやすさ、ユーザーの意図への適合性\n"
            "- 倫理的配慮や潜在的なバイアスの有無\n"
            "あなたの指摘は、最終的な完成版を作成するための最後の重要なフィードバックとなります。"
        )
        step6_prompt_for_claude_review2 = f"""ユーザーの現在の質問: 「{original_prompt}」
これまでの会話の主要な目的: 「{initial_user_prompt_for_session}」

複数のAIが協力して作成した「第1最終回答」(Gemini作成):
「{step5_res.response}」

あなたはこの「第1最終回答」を、上記のシステム指示に従って再度レビューし、最終的な改善提案をしてください。
"""
        step6_res = await get_claude_response(
            prompt_text=step6_prompt_for_claude_review2,
            system_instruction=step6_system_instruction_claude_review2,
            model="claude-3-opus-20240229",
            initial_user_prompt=initial_user_prompt_for_session
        )
        response_shell.step6_review2_claude = step6_res
        if step6_res.error or not step6_res.response:
            raise ValueError(f"Balance Mode - Step6 (Claudeレビュー2) 失敗: {step6_res.error or '応答なし'}")
        print(f"Claude レビュー2 (冒頭): {step6_res.response[:100].strip()}...")

        # ステップ7: OpenAI - 「第2最終回答（完成版）」の生成
        print("\nステップ7: OpenAIによる「第2最終回答（完成版）」生成中...")
        step7_system_prompt_openai_final = (
            f"特に、この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、"
            f"ユーザーの現在の具体的な質問は「{original_prompt}」であることを強く意識してください。\n"
            "これまでの全てのステップ（初期草案、レビュー1、改善案、追加情報、第1最終回答、レビュー2）の内容を総合的に判断し、"
            "特に最後のレビュー（ステップ6）での指摘事項を完全に解消するように、最高の最終回答を生成してください。"
        )
        step7_user_prompt_for_openai_final = f"""ユーザーの現在の質問: 「{original_prompt}」
会話全体の主要な目的: 「{initial_user_prompt_for_session}」

これまでのAIたちの議論の集大成です！
1.  AIによる初期回答草案 (OpenAI): 「{step1_res.response[:300].strip()}...」 (要約)
2.  上記へのレビュー1 (Claude): 「{step2_res.response[:300].strip()}...」 (要約)
3.  レビュー反映改善案 (Cohere): 「{step3_res.response[:300].strip()}...」 (要約)
4.  追加情報 (Perplexity): 「{(step4_res.response[:300].strip() + '...') if step4_res.response else '追加情報なし'}」 (要約)
5.  第1最終回答 (Gemini): 「{step5_res.response}」
6.  上記「第1最終回答」への最終レビュー (Claude): 「{step6_res.response}」

上記の全情報を踏まえ、特にClaudeによる最終レビュー（項目6）の指摘を全て解消し、指定された父親的で情報満載の口調で、ユーザーの現在の質問に対する完璧な最終回答を作成してください。
"""
        step7_res = await get_openai_response(
            prompt_text=step7_user_prompt_for_openai_final,
            system_role_description=step7_system_prompt_openai_final,
            model="gpt-4o",
            # 最終生成なので、ここまでの履歴ではなく、指示プロンプトに集約した情報を重視。
            # 必要であれば current_chat_history_for_this_turn を渡すことも可能。
            initial_user_prompt=initial_user_prompt_for_session
        )
        response_shell.step7_final_answer_v2_openai = step7_res # エラーの場合もそのまま格納
        if step7_res.error or not step7_res.response:
            # 最終ステップでエラーが起きても、ここまでの情報を返すため、ここでは敢えて ValueError を raise しない。
            # overall_error には記録せず、step7_final_answer_v2_openai.error にエラー情報が含まれる。
            print(f"Balance Mode - Step7 (OpenAI最終回答) でエラーまたは応答なし: {step7_res.error or '応答なし'}")
            # フォールバックとして、ステップ5の回答を最終回答とすることも検討できる。
            # if response_shell.step5_final_answer_gemini and response_shell.step5_final_answer_gemini.response:
            #     response_shell.step7_final_answer_v2_openai = response_shell.step5_final_answer_gemini
            #     print("警告: ステップ7でエラーのため、ステップ5の回答を最終回答としました。")
            # else:
            #     # ステップ5もなければ、エラーメッセージを最終回答とする
            #     response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            #         source="Fallback Error",
            #         response="申し訳ありません。最終回答の生成中に問題が発生しました。",
            #         error=step7_res.error or "ステップ7で不明なエラー"
            #     )
            # 今回は、ステップ7の結果をそのまま格納する
        else:
            print(f"OpenAI 第2最終回答 (冒頭): {step7_res.response[:100].strip()}...")

        print("--- バランスモード終了 ---")

    except ValueError as e:
        error_message = f"バランスモードの処理中にエラーが発生しました: {str(e)}"
        print(error_message)
        response_shell.overall_error = error_message
        # エラーが発生したステップまでの結果は response_shell に格納されている
        # 必要であれば、エラー時の最終応答を response_shell.step7_final_answer_v2_openai に設定
    if not response_shell.step7_final_answer_v2_openai:
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse( # <<< schemas.
            source="Code Mode Error Step",
            # ...
        )
    
    print(f"run_balance_mode_flow が返却する response_shell の内容 (JSON): {response_shell.model_dump_json(indent=2) if response_shell else 'None'}")
    return response_shell


import re

def format_code_mode_output(code: str, explanation: str, tests: str) -> str:
    # コード部分の前後に余計な```がないか除去（AI出力にはよく含まれる）
    code_clean = re.sub(r"^```[a-z]*\n?|\n?```$", "", code.strip(), flags=re.IGNORECASE)
    tests_clean = re.sub(r"^```[a-z]*\n?|\n?```$", "", tests.strip(), flags=re.IGNORECASE)
    explanation_clean = explanation.strip()

    return (
        "## 生成されたコード\n"
        "このコードをそのまま使えます。下記をコピペしてください。\n\n"
        "```python\n"
        f"{code_clean}\n"
        "```\n\n"
        "## コードの解説\n"
        f"{explanation_clean}\n\n"
        "## テストケース例\n"
        "```python\n"
        f"{tests_clean}\n"
        "```\n"
    )





async def run_search_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str]
) -> schemas.CollaborativeResponseV2:
    print("\n--- 新・検索特化モード（ペイルの知恵）開始 ---")
    # response_shell.search_fragments は [] で初期化されている想定 (pydanticのdefault_factory)
    response_shell.search_summary_text = None # 初期化
    response_shell.step7_final_answer_v2_openai = None # 初期化

    # このターンでAIに渡す完全な会話履歴 (モードフロー内で必要に応じて使用)
    current_chat_history_with_prompt = list(chat_history_for_ai) # コピー
    current_chat_history_with_prompt.append({"role": "user", "content": original_prompt})

    collected_fragments: List[schemas.IndividualAIResponse] = []

    try:
        # --- ステップ1: 最初のクエリでPerplexity検索 ---
        print("検索特化モード ステップ1: Perplexityによる初期検索中...")
        perplexity_initial_prompt = (
            f"ユーザーリクエスト「{original_prompt}」について、関連する情報を幅広く検索してください。\n"
            f"特に、ウェブページ、ニュース記事、SNS投稿、レビュー、学術論文の要旨など、多様な情報源からの情報を探してください。\n"
            f"見つかった各情報については、「可能な限り原文のまま、一切省略せずに」その内容をリストアップしてください。\n"
            f"さらに、各情報には、取得可能な範囲で「出典URL」「発行日」「著者名」「コンテンツの種類（例：ニュース記事、SNS投稿など）」を必ず付記してください。\n"
            f"情報の質と量を重視し、詳細なデータ収集を目的とします。"
        )
        step1_res_perplexity = await get_perplexity_response(
            prompt_for_perplexity=perplexity_initial_prompt,
            model="sonar-pro" # または "sonar-reasoning-pro"
        )
        if step1_res_perplexity.response:
            # Perplexityの応答をパースしてメタデータを付与する処理が必要 (ここでは簡略化)
            # 実際のPerplexity APIの応答形式に合わせてパース処理を実装してください。
            # 以下は仮のメタデータ付与です。
            collected_fragments.append(schemas.IndividualAIResponse(
                source="Perplexity 初期検索",
                response=step1_res_perplexity.response,
                links=step1_res_perplexity.links,
                query=original_prompt,
                intent="初期情報収集",
                source_url=step1_res_perplexity.links[0] if step1_res_perplexity.links else None,
                content_type="search_result_summary", # Perplexityは多様なソースを返す
                # published_date, author などもAPI応答から抽出できれば設定
            ))
        if step1_res_perplexity.error:
            print(f"ステップ1 Perplexity検索エラー: {step1_res_perplexity.error}")
            collected_fragments.append(schemas.IndividualAIResponse(
                source="Perplexity 初期検索エラー", error=step1_res_perplexity.error, query=original_prompt
            ))

        # --- ステップ2: Claude（またはGPT-4）で“抜けている観点”を自動抽出 ---
        print("\n検索特化モード ステップ2: 不足観点の抽出中...")
        missing_aspects_list: List[str] = []
        if not collected_fragments or not collected_fragments[-1].response:
            print("ステップ1の検索結果がないため、不足観点抽出をスキップします。")
        else:
            initial_search_results_text = "\n\n".join([f.response for f in collected_fragments if f.response])

            missing_aspects_system_prompt = ( # キャラ口調なし、検索モード専用マーカー付与
                "PALEAI_SEARCH_FORMATTING_TASK_MARKER\n" # このマーカーでAIヘルパー側でキャラ口調を抑制
                "あなたは、与えられた情報群を分析し、まだ触れられていない重要な論点、深掘りすべき追加情報、あるいは異なる視点を的確に見つけ出す専門家です。\n"
                "提案する観点は、次の検索ステップで具体的な検索クエリとして使用できるような、簡潔かつ明確な言葉で表現してください。\n"
                "出力は、提案される検索クエリのリストのみとし、各クエリは20字程度に収め、最大で5つまでとしてください。説明や前置きは一切不要です。"
            )
            missing_aspects_user_prompt_claude = (
                f"ユーザーの当初のリクエストは「{original_prompt}」です。\n"
                f"以下に、このリクエストに対する初期検索で得られた情報群を示します。\n\n"
                f"--- 初期検索結果群 ---\n{initial_search_results_text}\n--- 初期検索結果群ここまで ---\n\n"
                f"これらの情報を踏まえ、ユーザーがさらに知りたいであろう「まだカバーされていない重要な論点」や「深掘りすべき追加のキーワードや質問」を、"
                f"具体的な検索クエリとしてリスト形式で提案してください。提案は最大5つまで、各20字程度でお願いします。リスト以外の説明文は不要です。"
            )
            step2_res_claude = await get_claude_response( # または get_openai_response
                prompt_text=missing_aspects_user_prompt_claude,
                system_instruction=missing_aspects_system_prompt, # 専用システムプロンプト
                model="claude-3-haiku-20240307", # Opusより高速・安価なモデルで十分な場合も
                # chat_history=current_chat_history_with_prompt, # 必要に応じて文脈として渡す
                initial_user_prompt=initial_user_prompt_for_session
            )
            if step2_res_claude.response:
                collected_fragments.append(schemas.IndividualAIResponse(
                    source="Claude 不足観点抽出", response=step2_res_claude.response, intent="不足観点リスト"
                ))
                # クエリリストをパース (AIの出力形式に依存するため、適宜調整が必要)
                missing_aspects_list = [q.strip().lstrip("-・* ").rstrip("?") for q in step2_res_claude.response.splitlines() if q.strip() and len(q.strip()) > 3] # 簡単なフィルタ
            if step2_res_claude.error:
                print(f"ステップ2 不足観点抽出エラー: {step2_res_claude.error}")
                collected_fragments.append(schemas.IndividualAIResponse(
                    source="Claude 不足観点抽出エラー", error=step2_res_claude.error
                ))

        # --- ステップ3: 抜け観点ごとに再びPerplexity（またはGemini Web/News）で深掘り追加検索 ---
        print(f"\n検索特化モード ステップ3: {len(missing_aspects_list)}件の不足観点で追加検索中...")
        for i, aspect_query in enumerate(missing_aspects_list):
            if not aspect_query: continue
            print(f"  追加検索 {i+1}/{len(missing_aspects_list)}: 「{aspect_query}」")
            perplexity_followup_prompt = (
                f"ユーザーの当初のリクエストは「{original_prompt}」でした。\n"
                f"それに関連する深掘り観点として「{aspect_query}」が特定されました。\n"
                f"この「{aspect_query}」について、関連情報を幅広く検索し、ウェブページ、ニュース記事、SNS投稿、レビュー、論文の要旨など、見つかった情報を「可能な限り原文のまま、省略せずに」リストアップしてください。\n"
                f"各情報には、出典URL、発行日、著者名、コンテンツの種類を可能な限り付記してください。"
            )
            # Perplexityを使うが、GeminiのWeb検索機能や他の検索APIも検討可能
            step3_sub_res_perplexity = await get_perplexity_response(
                prompt_for_perplexity=perplexity_followup_prompt, model="sonar-pro"
            )
            if step3_sub_res_perplexity.response:
                collected_fragments.append(schemas.IndividualAIResponse(
                    source=f"Perplexity 追加検索 ({aspect_query[:20]}...)",
                    response=step3_sub_res_perplexity.response,
                    links=step3_sub_res_perplexity.links,
                    query=aspect_query,
                    intent=f"深掘り ({aspect_query[:20]}...)",
                    source_url=step3_sub_res_perplexity.links[0] if step3_sub_res_perplexity.links else None,
                    content_type="search_result_summary_followup"
                    # published_date, author などもAPI応答から抽出できれば設定
                ))
            if step3_sub_res_perplexity.error:
                print(f"  追加検索「{aspect_query}」エラー: {step3_sub_res_perplexity.error}")
                collected_fragments.append(schemas.IndividualAIResponse(
                    source=f"Perplexity 追加検索エラー ({aspect_query[:20]}...)",
                    error=step3_sub_res_perplexity.error, query=aspect_query
                ))

        # --- ステップ4: 全取得情報をまとめて、Gemini AdvancedまたはGPT-4に成形・分類だけさせる ---
        print("\n検索特化モード ステップ4: 全情報の整形・分類中...")
        if not any(f.response for f in collected_fragments): # 有効な情報断片がない場合
            response_shell.overall_error = "検索結果がありませんでした。"
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Search Mode Orchestrator", error="情報収集ステップで有効な結果が得られませんでした。"
            )
            response_shell.search_fragments = collected_fragments # エラー情報だけでも返す
            return response_shell

        all_fragments_text_for_formatting = ""
        for idx, frag in enumerate(collected_fragments):
            if frag.response: # 有効な応答のみを整形対象とする
                frag_header = f"--- 情報断片 {idx+1}: (ソース: {frag.source}, クエリ: {frag.query or 'N/A'}, 意図: {frag.intent or 'N/A'}) ---\n"
                frag_content = frag.response
                frag_metadata_lines = []
                if frag.source_url: frag_metadata_lines.append(f"  - 出典URL: {frag.source_url}")
                if frag.published_date: frag_metadata_lines.append(f"  - 発行日: {frag.published_date}")
                if frag.author: frag_metadata_lines.append(f"  - 著者/発信者: {frag.author}")
                if frag.content_type: frag_metadata_lines.append(f"  - コンテンツ種類: {frag.content_type}")
                frag_links_text = "\n".join([f"  - 関連リンク: {link}" for link in frag.links]) if frag.links and not frag.source_url else "" # source_urlと重複する可能性があるので注意

                all_fragments_text_for_formatting += f"{frag_header}{frag_content}\n"
                if frag_metadata_lines: all_fragments_text_for_formatting += "\n".join(frag_metadata_lines) + "\n"
                if frag_links_text: all_fragments_text_for_formatting += frag_links_text + "\n"
                all_fragments_text_for_formatting += "\n\n"

        if not all_fragments_text_for_formatting.strip():
             response_shell.overall_error = "整形対象となる有効な検索情報がありませんでした。"
             response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(source="Search Orchestrator", error="有効な情報断片なし")
             response_shell.search_fragments = collected_fragments
             return response_shell

        # 整形用システムプロンプト (キャラクター性抑制マーカー付き)
        formatting_system_prompt_gemini = (
            "PALEAI_SEARCH_FORMATTING_TASK_MARKER\n"
            "あなたは、提供された複数の情報断片を、ユーザーが理解しやすいように整理・構造化する専門のAI編集者です。\n"
            "あなたのタスクは以下の通りです。\n"
            "1. **情報断片の完全な提示**: 提供された各情報断片（原文抜粋、検索結果の応答、ニュース記事の断片、SNS投稿のテキスト、レビュー内容など）を、「一切省略・圧縮・要約せず」、その内容全文を保持してください。\n"
            "2. **構造化とメタデータ付与**: 各情報断片について、以下の情報を可能な限り抽出し、明記してください。\n"
            "    - 元の情報源 (例: Perplexity 初期検索, Claude 不足観点抽出 など、入力で提供されたソース名)\n"
            "    - 元の検索クエリ (もしあれば)\n"
            "    - 情報収集の意図 (もしあれば)\n"
            "    - 出典URL (もしあれば)\n"
            "    - 発行日 (もしあれば)\n"
            "    - 著者/発信者 (もしあれば)\n"
            "    - コンテンツの種類 (もしあれば)\n"
            "3. **分類とグループ化**: 上記のメタデータを付与した上で、収集された全情報断片を、内容の関連性やトピック、情報源の種類など、最も適切と思われる観点からグループ化し、見出しや区切り線などを用いて視覚的に分かりやすく構造化して提示してください。あなたの解釈や意見、追加情報は一切含めないでください。\n"
            "4. **最終まとめの生成**: 全ての構造化された情報断片を提示した後、**必ず最後に**「【最終まとめ】」という見出しを付け、その後に1～2段落程度の非常に短い概要（収集された情報のハイライトや傾向など）を記述してください。このまとめ部分以外では、絶対に情報を要約したり、個人的な意見を述べたりしないでください。\n"
            "出力は、まず構造化された情報断片のリスト、その次に「【最終まとめ】」とそれに続く短い概要、という形式にしてください。"
        )
        formatting_user_prompt_gemini = (
            f"ユーザーの当初の質問は「{original_prompt}」です。\n"
            f"以下に、この質問に関連して収集された複数の情報断片（原文抜粋、検索結果の応答、ニュース記事の断片、SNS投稿のテキスト、レビュー内容など）と、それらに付随するメタデータ（出典URL、発行日、著者、コンテンツ種類など）があります。\n\n"
            f"--- 全情報断片ここから ---\n{all_fragments_text_for_formatting.strip()}\n--- 全情報断片ここまで ---\n\n"
            f"上記のシステム指示に従い、これらの情報断片とメタデータを一切省略・要約せず、分野・時系列・ソースの種類別などに適切に分類・整形し、提示してください。\n"
            f"各断片には、収集時の情報（ソース、クエリ、意図、URL、日付、著者など）を可能な限り明記してください。\n"
            f"最後に1～2段落の「超短いまとめ」だけを「【最終まとめ】」という見出しで付与してください。"
        )

        step4_formatting_res_gemini = await get_gemini_response( # または get_openai_response("gpt-4o")
            prompt_text=formatting_user_prompt_gemini,
            system_instruction=formatting_system_prompt_gemini, # キャラ抑制マーカー含む
            model_name="gemini-1.5-pro-latest", # 高度な長文処理能力を持つモデル推奨
            initial_user_prompt=initial_user_prompt_for_session # 文脈として渡す
        )

        if step4_formatting_res_gemini.response:
            summary_marker = "【最終まとめ】"
            formatted_fragments_display = step4_formatting_res_gemini.response
            summary_text = "（AIによる自動まとめはありませんでした）" # デフォルト

            if summary_marker in formatted_fragments_display:
                parts = formatted_fragments_display.split(summary_marker, 1)
                formatted_fragments_display = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    summary_text = parts[1].strip()
                else: # マーカーはあるが中身がない場合
                    summary_text = "まとめ部分が空でした。"
            else:
                print("警告: 整形結果に【最終まとめ】マーカーが見つかりませんでした。応答全体を断片情報として扱います。")

            # 整形されたメインコンテンツ (断片リスト) を格納
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Gemini 整形・分類結果",
                response=formatted_fragments_display,
                query=original_prompt, # この整形タスクの元のトリガー
                intent="全情報整形済みリスト"
            )
            response_shell.search_summary_text = summary_text # 分離したまとめ
        else:
            response_shell.overall_error = "最終的な情報の整形・分類に失敗しました。"
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Search Mode Orchestrator", error=f"最終整形ステップでエラー: {step4_formatting_res_gemini.error or '応答なし'}"
            )

        # 生の断片情報もレスポンスに含める
        response_shell.search_fragments = collected_fragments
        print("--- 新・検索特化モード（ペイルの知恵）終了 ---")

    except ValueError as ve:
        error_message = f"検索特化モードの処理中にエラー: {str(ve)}"
        print(error_message)
        response_shell.overall_error = error_message
        response_shell.search_fragments = collected_fragments # ここまでの断片は返す
        if not response_shell.step7_final_answer_v2_openai:
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Search Mode Error", error=str(ve)
            )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_message = f"検索特化モードで予期せぬエラー: {str(e)}"
        print(f"{error_message}\nTrace: {error_trace}")
        response_shell.overall_error = "サーバー内部で予期せぬエラーが発生しました（検索特化）。"
        response_shell.search_fragments = collected_fragments
        if not response_shell.step7_final_answer_v2_openai:
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Search Mode Unexpected Error", error=str(e)
            )

    return response_shell
  
    print("\n--- バランスモード開始 ---")

    step1_res = schemas.IndividualAIResponse(source="OpenAI (Initial Draft)", error="未実行")
    step2_res = schemas.IndividualAIResponse(source="Claude (Review 1)", error="未実行")
    step3_res = schemas.IndividualAIResponse(source="Cohere (Improved Draft)", error="未実行")
    step4_res = schemas.IndividualAIResponse(source="PerplexityAI (Fact Check/Research)", error="未実行")
    step5_res = schemas.IndividualAIResponse(source="Gemini (Synthesized Answer 1)", error="未実行")
    step6_res = schemas.IndividualAIResponse(source="Claude (Review 2)", error="未実行")
    step7_res = schemas.IndividualAIResponse(source="OpenAI (Final Polished Answer)", error="未実行")
        
    try:
        # ステップ1: OpenAI - 初期回答の草案生成
        print("ステップ1: OpenAIによる初期回答生成中...")
        step1_system_prompt = "あなたは、与えられた質問とこれまでの会話履歴を考慮し、まずは網羅的かつ客観的な情報に基づいた基本的な回答の草案を、序論・本論（複数の主要点）・結論の形式で構造化して作成するAIアシスタントです。"
        step1_res = await get_openai_response( # ★★★ get_openai_response を呼び出し ★★★
            prompt_text=original_prompt, # get_openai_response内で履歴があればそちらを優先する想定
            system_role_description=step1_system_prompt,
            chat_history=chat_history_for_ai # ★★★ チャット履歴を渡す ★★★
        )
        response_shell.step1_initial_draft_openai = step1_res
        if step1_res.error or not step1_res.response:
            raise ValueError(f"Balance Mode - Step1 Error: {step1_res.error or '応答なし'}")
        print(f"OpenAI 回答草案 (冒頭): {step1_res.response[:100].strip()}...")

        # ステップ2: Claude - 初期回答の批判的レビュー
        # レビュータスクでは、レビュー対象のテキストと元の質問が重要。全履歴は必ずしも渡さない。
        print("\nステップ2: Claudeによる批判的レビュー中...")
        step2_system_instruction_claude = "あなたはAIの回答を非常に厳しく、批判的に評価する専門のレビュアーです。論理の飛躍、根拠の薄弱さ、情報の偏り、表現の曖昧さなどを重点的にチェックし、改善すべき点を具体的かつ実行可能な形で【番号付きリスト形式で5つ以上】提案してください。"
        step2_prompt_for_claude = f"""以下のユーザープロンプトと、それに対するAIの初期回答の草案があります。
ユーザープロンプト: 「{original_prompt}」
AIの初期回答草案 (OpenAIより):
「{step1_res.response}」

この初期回答草案をレビューし、改善提案をしてください。あなたのレビューは、次のAIがこの草案を大幅に改善するための重要な土台となります。
"""
        step2_res = await get_claude_response( # ★★★ get_claude_response を呼び出し ★★★
            prompt_text=step2_prompt_for_claude, # レビュー対象のプロンプト
            system_instruction=step2_system_instruction_claude
            # chat_history=None # ここでは履歴を渡さない例（ステップ1の出力に集中するため）
        )
        response_shell.step2_review_claude = step2_res
        if step2_res.error or not step2_res.response:
            raise ValueError(f"Balance Mode - Step2 Error: {step2_res.error or '応答なし'}")
        print(f"Claude レビュー1 (冒頭): {step2_res.response[:100].strip()}...")

        # ステップ3: Cohere - 第1改善案の作成
        print("\nステップ3: Cohereによる第1改善案作成中...")
        step3_preamble_cohere = "あなたは、与えられた草案と詳細なレビューを元に、文章を大幅に改善し、より論理的で説得力のある内容に再構築する編集専門のAIです。レビューでの指摘事項を具体的に反映させ、文章全体の質を向上させてください。"
        step3_prompt_for_cohere = f"""以下のユーザープロンプト、AIによる初期回答草案、そしてその草案への詳細な批判的レビューがあります。
ユーザープロンプト: 「{original_prompt}」
初期回答草案 (OpenAIより):
「{step1_res.response}」
批判的レビューと改善提案 (Claudeより):
「{step2_res.response}」

これらの情報をすべて考慮し、初期回答草案をレビューでの指摘事項【すべて】に基づいて具体的に修正・拡張し、全面的に書き直した「第1改善案」を作成してください。
"""
        step3_res = await get_cohere_response( # ★★★ get_cohere_response を呼び出し ★★★
            prompt_text=step3_prompt_for_cohere,
            preamble=step3_preamble_cohere
        )
        response_shell.step3_improved_draft_cohere = step3_res
        if step3_res.error or not step3_res.response:
            raise ValueError(f"Balance Mode - Step3 Error: {step3_res.error or '応答なし'}")
        print(f"Cohere 第1改善案 (冒頭): {step3_res.response[:100].strip()}...")

        # ステップ4: Perplexity AI - Claudeのレビューに基づく具体的な情報収集と提示
        print("\nステップ4: Perplexity AIによる情報収集と提示中...")
        step4_prompt_for_perplexity = f"""ユーザーの元の質問は「{original_prompt}」です。
これに関連して、以前のAIの回答草案に対し、別のAI（Claude）が以下のレビューを行いました。

Claudeによるレビューと改善提案:
「{step2_res.response}」

あなたはこのレビューを精読し、特にClaudeが指摘している【情報の不足箇所】、【根拠が弱いとされた部分】、【具体例が求められている点】について、信頼できる最新のウェブ情報を基に調査し、その結果をまとめてください。
あなたの提供する情報は、次のステップでAIがより質の高い回答を作成するための『事実に基づいた参考資料』となります。可能であれば、情報源のURLも示してください。
"""
        step4_res = await get_perplexity_response( # ★★★ get_perplexity_response を呼び出し ★★★
            prompt_for_perplexity=step4_prompt_for_perplexity,
            model="sonar-reasoning-pro" # Perplexityのモデル名を指定
        )
        response_shell.step4_comprehensive_answer_perplexity = step4_res
        if step4_res.error or (step4_res.response is not None and len(step4_res.response.strip()) < 5 and not step4_res.error):
            print(f"Perplexity AIからの応答が不十分またはエラー: {step4_res.error or '応答が短すぎます/ありません'}")
            step4_res.response = step4_res.response or ""
        elif step4_res.response:
            print(f"Perplexity AI リサーチ結果 (冒頭): {step4_res.response[:100].strip()}...")
        else:
            step4_res.response = ""
            print("Perplexity AIから応答がありませんでした（エラーもなし）。")

        # ステップ5: Gemini - 「第1最終回答」の統合・編集
        print("\nステップ5: Geminiによる「第1最終回答」生成中...")
        step5_system_instruction_gemini = (
            "あなたは、ユーザーからの質問に対し、複数のAIによる多角的な検討とリサーチを経て、"
            "質の高い回答を構築するAI編集長です。"
            "提供された全ての情報を慎重に吟味し、矛盾を解消し、読者にとって最も価値のある最終的なベストアンサーを構築してください。"
            "批判的思考と高度な編集能力を発揮し、論理的で一貫性のある、正確かつ詳細で網羅的な、分かりやすい文章にまとめてください。"
        )
        step5_prompt_for_gemini = f"""以下の情報が提供されています。
1.  ユーザーの元の質問: 「{original_prompt}」
2.  AIによる初期回答草案 (OpenAI作成): 「{step1_res.response}」
3.  初期回答草案への批判的レビュー (Claude作成): 「{step2_res.response}」
4.  レビューを反映し、改善された第1改善案 (Cohere作成): 「{step3_res.response}」
5.  Claudeのレビューに基づき収集された事実情報とデータ集 (Perplexity AI作成): 「{step4_res.response if step4_res.response else "Perplexityからの情報提供はありませんでした。"}」

これらの情報をすべて活用し、元のユーザープロンプトに対する「第1最終回答」を生成してください。
Claudeのレビューでの指摘を解消するために、Perplexity AIが提供した情報（もしあれば）を参考にし、Cohereの改善案を土台としてください。
"""
        step5_res = await get_gemini_response( # ★★★ get_gemini_response を呼び出し ★★★
            prompt_text=step5_prompt_for_gemini,
            system_instruction=step5_system_instruction_gemini
        )
        response_shell.step5_final_answer_gemini = step5_res
        if step5_res.error or not step5_res.response:
            raise ValueError(f"Balance Mode - Step5 Error: {step5_res.error or '応答なし'}")
        print(f"Gemini 第1最終回答 (冒頭): {step5_res.response[:100].strip()}...")

        # ステップ6: Claude - 「第1最終回答」の再レビュー
        print("\nステップ6: Claudeによる「第1最終回答」の再レビュー中...")
        # ... (step6_prompt_for_claude_review2 の作成は既存のまま) ...
        step6_system_instruction_claude_review2 = (
            "あなたはAIが生成した高度な回答をさらに磨き上げるための最終レビューを行う、超一流の編集・校閲AIです。\n"
            "提示された「第1最終回答」を、さらに高いレベルの品質に引き上げるために、非常に厳しく、多角的な視点から再度レビューしてください。\n"
            "特に以下の点に注目し、具体的な改善点を【番号付きリスト形式で3つ以上】挙げてください。\n"
            "- 論理の一貫性と飛躍のなさ\n"
            "- 情報の深さと洞察力\n"
            "- 表現の洗練度と説得力\n"
            "- 読者への訴求力と分かりやすさ\n"
            "- 倫理的配慮や潜在的なバイアスの有無\n"
            "あなたの指摘は、最終的な完成版を作成するための最後の重要なフィードバックとなります。"
        )
        step6_prompt_for_claude_review2 = f"""以下は、ユーザーの元の質問「{original_prompt}」に対して、複数のAIが協力して作成した「第1最終回答」です。
第1最終回答 (Gemini作成):
「{step5_res.response}」

あなたはこの「第1最終回答」を、上記のシステム指示に従って再度レビューしてください。
"""
        step6_res = await get_claude_response( # ★★★ get_claude_response を呼び出し ★★★
            prompt_text=step6_prompt_for_claude_review2,
            system_instruction=step6_system_instruction_claude_review2
        )
        response_shell.step6_review2_claude = step6_res
        if step6_res.error or not step6_res.response:
            raise ValueError(f"Balance Mode - Step6 Error: {step6_res.error or '応答なし'}")
        print(f"Claude レビュー2 (冒頭): {step6_res.response[:100].strip()}...")

        # ステップ7: OpenAI - 「第2最終回答（完成版）」の生成
        print("\nステップ7: OpenAIによる「第2最終回答（完成版）」生成中...")
        # ... (step7_user_prompt_for_openai_final の作成は既存のまま) ...
        step7_user_prompt_for_openai_final = f"""お疲れ様！いよいよ最終仕上げだよ。
ユーザーさんの最初の質問は「{original_prompt}」だったよね。
これまでのAIたちの頑張りを見てみよう！

Geminiくんが「第1最終回答」としてこんな風にまとめてくれたんだ：
「{step5_res.response if step5_res and step5_res.response else 'Geminiからの第1最終回答はありませんでした。'}」

そしたら、頼れるClaudeさんが、この「第1最終回答」にこんな最終レビューと改善提案をくれたよ：
「{step6_res.response if step6_res and step6_res.response else 'Claudeからの最終レビューはありませんでした。'}」

さあ、君の出番だ！これらの情報をぜーんぶ参考にして、Claudeさんの指摘もバッチリ反映させてね。
「第1最終回答」を元にしつつ、もっともっと分かりやすくて、面白くて、情報量たっぷりでボリューム満点の詳しい説明で、ユーザーさんが「なるほどー！めっちゃ分かりやすいし、詳しい！」って感動するような「完成版」の回答を作ってほしいんだ。
細かい点や背景、具体例もたくさん入れて、読み応えのある回答をよろしくね！🎉
"""
        step7_res = await get_openai_response( # ★★★ get_openai_response を呼び出し ★★★
            prompt_text=step7_user_prompt_for_openai_final,
            system_role_description=None,
            model="gpt-4o"
            # chat_history=None # ここでは集約情報から最終回答を生成するため、全履歴は渡さない例
        )
        if step7_res.error or not step7_res.response:
            raise ValueError(f"Balance Mode - Step7 Error: {step7_res.error or '応答なし'}")
        # print(f"OpenAI 第2最終回答 (冒頭): {step7_res.response[:100].strip()}...") # この行は有効なはず

        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse( # <<< schemas. を追加
            source=step7_res.source,
            response=step7_res.response,
            error=step7_res.error
        )
        print("--- バランスモード終了 ---")

    except ValueError as e:
        error_message = f"バランスモードの処理中にエラーが発生しました: {str(e)}"
        print(error_message)
        response_shell.overall_error = error_message
        # エラーが発生した場合のフォールバックも schemas. を付ける
        if not response_shell.step7_final_answer_v2_openai or \
           (response_shell.step7_final_answer_v2_openai.error == "未実行" and not response_shell.step7_final_answer_v2_openai.response): # type: ignore
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse( # <<< schemas. を追加
                source="Balance Mode Error Step",
                error=str(e),
                response=f"処理中にエラーが発生しました: {str(e)}"
            )
    
    print(f"run_balance_mode_flow が返却する response_shell の内容 (JSON): {response_shell.model_dump_json(indent=2) if response_shell else 'None'}")
    return response_shell


# 修正後
async def run_code_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str]
) -> schemas.CollaborativeResponseV2:

    print("\n--- コード生成特化モード開始 ---")
    steps_executed: List[schemas.IndividualAIResponse] = []
    refined_requirements = ""
    detailed_specs_and_pseudo = ""
    generated_code_v1 = ""
    code_review_feedback = ""
    generated_code_v2 = ""
    test_cases_suggestion = ""
    code_explanation = ""

    current_chat_history_for_this_turn = list(chat_history_for_ai)
    current_chat_history_for_this_turn.append({"role": "user", "content": original_prompt})
    print(f"Code Mode: このターンでAIに渡す完全な履歴は {len(current_chat_history_for_this_turn)} 件")

    try:
        # ステップC0: プロンプトの精密化
        print("コードモード ステップC0: ユーザープロンプトの精密化中...")
        c0_system_instruction = (
            "あなたはユーザーの曖昧なコード生成リクエストを分析し、後続のAIが具体的なコードを生成するために必要な情報を明確にする専門家です。"
            "不足している情報があれば、ユーザーに確認を促すような質問形式で補足することも考慮してください。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」です。"
        )
        c0_user_prompt = (
            f"現在のユーザーリクエストは「{original_prompt}」です。\n"
            "このリクエストに基づいてコードを生成するために必要な情報を、以下の項目でできる限り詳細に、かつ構造化して整理してください。\n"
            # ... (C0の整理項目は変更なし) ...
        )
        c0_res = await get_gemini_response(
            prompt_text=c0_user_prompt, # ユーザーへの質問形式のプロンプト
            system_instruction=c0_system_instruction,
            model_name="gemini-1.5-pro-latest", # または "gemini-1.5-flash-latest"
            chat_history=list(current_chat_history_for_this_turn), # ここまでの全履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c0_res)
        if c0_res.error or not c0_res.response:
            raise ValueError(f"コードモード ステップC0 (プロンプト精密化) 失敗: {c0_res.error or '応答がありませんでした。'}")
        refined_requirements = c0_res.response
        print(f"ステップC0 - 精密化された要件:\n{refined_requirements[:300].strip()}...")

        # ステップC1: 詳細仕様の策定と疑似コード/ロジック設計
        print("\nコードモード ステップC1: 詳細仕様と疑似コード策定中...")
        c1_system_instruction = (
            "あなたは経験豊富なソフトウェアアーキテクトです。"
            "提示されたプログラム要件を基に、より詳細な技術仕様、主要な関数やモジュールの設計、そして中心となるアルゴリズムや処理フローの疑似コードを作成してください。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            "考慮すべきエッジケースや、適切なエラーハンドリングについても言及をお願いします。"
        )
        c1_user_prompt = (
            f"以下のプログラム要件（ステップC0で明確化されたもの）があります:\n--- 要件ここから ---\n{refined_requirements}\n--- 要件ここまで ---\n"
            "この要件を満たすための詳細な技術仕様、主要なロジックの設計（関数名や役割など）、そして中心となる処理の疑似コードを作成してください。"
        )
        c1_res = await get_claude_response(
            prompt_text=c1_user_prompt,
            system_instruction=c1_system_instruction,
            model="claude-3-opus-20240229",
            chat_history=list(current_chat_history_for_this_turn), # C0の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c1_res)
        if c1_res.error or not c1_res.response:
            raise ValueError(f"コードモード ステップC1 (詳細仕様策定) 失敗: {c1_res.error or '応答がありませんでした。'}")
        detailed_specs_and_pseudo = c1_res.response
        print(f"ステップC1 - 詳細仕様と疑似コード (冒頭):\n{detailed_specs_and_pseudo[:300].strip()}...")

        # ... (以降のステップ C2～C6 も同様に、各AIヘルパー関数呼び出し時に chat_history と initial_user_prompt を渡し、
        #     システムプロンプトや指示プロンプトにこれらの文脈情報を適切に組み込む) ...

        # ステップC2: 第1コード生成
        print("\nコードモード ステップC2: 第1コード生成中...")
        c2_system_role_description = (
            "あなたは、提示された詳細な技術仕様と疑似コードに基づいて、高品質で、読みやすく、コメントが適切に付与されたPythonコードを生成する熟練のプログラマーです。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            "仕様に沿って、効率的で堅牢なコードを作成してください。"
        )
        c2_user_prompt = (
            f"以下の詳細仕様と疑似コードに基づいて、Pythonでコードを生成してください。\n\n"
            f"--- 詳細仕様と疑似コードここから ---\n{detailed_specs_and_pseudo}\n--- 詳細仕様と疑似コードここまで ---\n\n"
            "生成するコードは、上記の仕様とロジックを正確に実装し、適切な変数名、関数名、そして理解を助けるコメントを含めてください。"
        )
        c2_res = await get_openai_response(
            prompt_text=c2_user_prompt,
            system_role_description=c2_system_role_description,
            model="gpt-4o",
            chat_history=list(current_chat_history_for_this_turn), # C1の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c2_res)
        if c2_res.error or not c2_res.response:
            raise ValueError(f"コードモード ステップC2 (第1コード生成) 失敗: {c2_res.error or '応答がありませんでした。'}")
        generated_code_v1 = c2_res.response
        print(f"ステップC2 - 生成されたコード V1 (冒頭):\n{generated_code_v1[:300].strip()}...")

        # ステップC3: コードレビュー
        print("\nコードモード ステップC3: コードレビュー中...")
        c3_system_instruction = (
            "あなたは非常に経験豊富で、細部まで注意深くコードをレビューするシニアソフトウェアエンジニアです。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            "提示されたPythonコードを精査し、具体的な改善提案をリスト形式で挙げてください。"
            # ... (レビュー観点は省略) ...
        )
        c3_user_prompt = (
            f"以下のPythonコードについて、詳細なレビューと具体的な改善提案をお願いします。\n\n"
            f"--- コードここから ---\n{generated_code_v1}\n--- コードここまで ---\n"
        )
        c3_res = await get_claude_response(
            prompt_text=c3_user_prompt,
            system_instruction=c3_system_instruction,
            model="claude-3-opus-20240229",
            chat_history=list(current_chat_history_for_this_turn), # C2の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c3_res)
        if c3_res.error or not c3_res.response:
            print(f"コードモード ステップC3 (コードレビュー) でエラーまたは応答なし: {c3_res.error or '応答なし'}")
            code_review_feedback = "AIによるコードレビューはありませんでした。" 
        else:
            code_review_feedback = c3_res.response
        print(f"ステップC3 - コードレビュー結果 (冒頭):\n{code_review_feedback[:300].strip()}...")
        
        # ステップC4: 改善版コード生成
        print("\nコードモード ステップC4: 改善版コード生成中...")
        c4_system_role_description = (
            "あなたは、提示された元のPythonコードと、それに対する詳細なレビューおよび改善提案を理解し、レビューでの指摘事項を的確に反映させてコードを修正・改善する専門のプログラマーです。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            "元のコードの主要な機能は維持しつつ、品質を向上させてください。"
        )
        c4_user_prompt = (
            f"以下のオリジナルのPythonコードと、それに対するレビューおよび改善提案があります。\n\n"
            f"--- オリジナルコード (V1) ここから ---\n{generated_code_v1}\n--- オリジナルコード (V1) ここまで ---\n\n"
            f"--- レビューと改善提案 ここから ---\n{code_review_feedback}\n--- レビューと改善提案 ここまで ---\n\n"
            "上記のレビューと改善提案をすべて慎重に検討し、指摘された点を修正・反映した改善版のPythonコード（バージョン2）を生成してください。"
        )
        c4_res = await get_openai_response(
            prompt_text=c4_user_prompt,
            system_role_description=c4_system_role_description,
            model="gpt-4o",
            chat_history=list(current_chat_history_for_this_turn), # C3の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c4_res)
        if c4_res.error or not c4_res.response:
            raise ValueError(f"コードモード ステップC4 (改善版コード生成) 失敗: {c4_res.error or '応答がありませんでした。'}")
        generated_code_v2 = c4_res.response
        print(f"ステップC4 - 生成された改善版コード V2 (冒頭):\n{generated_code_v2[:300].strip()}...")

        # ステップC5: テストケース提案
        print("\nコードモード ステップC5: テストケース提案中...")
        c5_system_instruction = (
            "あなたは経験豊富なソフトウェアテストエンジニアです。提示されたPythonコードを分析し、その機能が正しく動作することを確認するためのテストケースを包括的に提案してください。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            # ... (テストケースの記述形式指示は省略) ...
        )
        c5_user_prompt = (
            f"以下のPythonコードについて、詳細なテストケースを提案してください。\n\n"
            f"--- 対象コード (バージョン2) ---\n{generated_code_v2}\n--- 対象コードここまで ---\n\n"
            "上記のシステム指示に従い、できるだけ多くの観点から、具体的で検証可能なテストケースを提案してください。"
        )
        c5_res = await get_claude_response(
            prompt_text=c5_user_prompt,
            system_instruction=c5_system_instruction,
            model="claude-3-opus-20240229",
            chat_history=list(current_chat_history_for_this_turn), # C4の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c5_res)
        if c5_res.error or not c5_res.response:
            print(f"コードモード ステップC5 (テストケース提案) でエラーまたは応答なし: {c5_res.error or '応答なし'}")
            test_cases_suggestion = "AIによるテストケースの提案はありませんでした。"
        else:
            test_cases_suggestion = c5_res.response
        print(f"ステップC5 - テストケース提案 (冒頭):\n{test_cases_suggestion[:300].strip()}...")

        # ステップC6: コードの使い方説明・解説
        print("\nコードモード ステップC6: コードの使い方説明・解説中...")
        c6_system_role_description = (
            "あなたは熟練したテクニカルライターであり、Pythonコードを初心者にも分かりやすく解説する専門家です。"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            "提示されたPythonコードについて、以下の点を網羅的に、かつ平易な言葉で説明してください。"
            # ... (解説項目指示は省略) ...
        )
        c6_user_prompt = (
            f"以下のPythonコードについて、詳細な使い方と解説をお願いします。\n\n"
            f"--- 対象コード (バージョン2) ---\n{generated_code_v2}\n--- 対象コードここまで ---\n\n"
            "上記のシステム指示に従い、このコードを初めて見る人でも理解しやすく、すぐに使えるような親切な解説を作成してください。"
        )
        c6_res = await get_openai_response(
            prompt_text=c6_user_prompt,
            system_role_description=c6_system_role_description,
            model="gpt-4o",
            chat_history=list(current_chat_history_for_this_turn), # C5の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(c6_res)
        if c6_res.error or not c6_res.response:
            print(f"コードモード ステップC6 (コード解説) でエラーまたは応答なし: {c6_res.error or '応答なし'}")
            code_explanation = "AIによるコードの解説はありませんでした。"
        else:
            code_explanation = c6_res.response
        print(f"ステップC6 - コード解説 (冒頭):\n{code_explanation[:300].strip()}...")
        
        final_code_mode_output = format_code_mode_output(
            generated_code_v2,
            code_explanation,
            test_cases_suggestion,
        )
        
        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Code Mode (Full Flow C0-C6)",
            response=final_code_mode_output
        )
        response_shell.code_mode_details = steps_executed
        print("--- コード生成特化モード (全ステップ完了) 終了 ---")

    except ValueError as e:
        error_message = f"コード生成特化モードの処理中にエラー: {str(e)}"
        print(error_message)
        response_shell.overall_error = error_message
        response_shell.code_mode_details = steps_executed
        if not response_shell.step7_final_answer_v2_openai: # エラー時に最終応答フィールドが空なら設定
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Code Mode Error Step", 
                error=str(e),
                response=f"申し訳ありません、コード生成処理中にエラーが発生しました。\nエラー内容: {str(e)}"
            )
    return response_shell

# 修正後
async def run_writing_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str]
) -> schemas.CollaborativeResponseV2:

    print("\n--- 執筆特化モード開始 ---")
    steps_executed: List[schemas.IndividualAIResponse] = []
    defined_requirements = ""
    finalized_requirements = ""
    article_outline = ""
    draft_content = ""
    review_and_suggestions = ""
    revised_draft_content = ""
    final_article_content = ""

    current_chat_history_for_this_turn = list(chat_history_for_ai)
    current_chat_history_for_this_turn.append({"role": "user", "content": original_prompt})
    print(f"Writing Mode: このターンでAIに渡す完全な履歴は {len(current_chat_history_for_this_turn)} 件")

    try:
        # ステップW0: 要件確認とテーマ深掘り
        print("執筆モード ステップW0: 要件確認とテーマ深掘り中...")
        w0_system_instruction = (
            "あなたはユーザーの多様な執筆リクエストを分析し、高品質なコンテンツを生成するために必要な構成要素を明確化する専門の編集アシスタントです。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」です。\n"
            # ... (W0のシステム指示の残りは変更なし) ...
        )
        w0_user_prompt = (
            f"ユーザーは以下の内容の執筆を希望しています。\nユーザーの執筆リクエスト: 「{original_prompt}」\n\n"
            # ... (W0のユーザープロンプトの残りは変更なし) ...
        )
        w0_res = await get_gemini_response(
            prompt_text=w0_user_prompt,
            system_instruction=w0_system_instruction,
            model_name="gemini-1.5-pro-latest",
            chat_history=list(current_chat_history_for_this_turn),
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w0_res)
        if w0_res.error or not w0_res.response:
            raise ValueError(f"執筆モード ステップW0 (要件確認) 失敗: {w0_res.error or '応答がありませんでした。'}")
        defined_requirements = w0_res.response
        print(f"ステップW0 - 定義された執筆要件 (冒頭):\n{defined_requirements[:300].strip()}...")
        
        # ... (finalized_requirements の作成は変更なし) ...
        supplemented_info = f"""
【開発者による補足情報（ステップW0のAIからの質問への一般的な回答方針）】
- 執筆物の種類: ユーザーの最初の執筆リクエスト「{original_prompt}」から最大限推測してください。不明確な場合は、最も可能性の高い種類を仮定するか、複数の可能性を提示してユーザーに選択を促すような形で進めてください。
- ターゲット読者: 「{original_prompt}」から推測される読者層、または執筆物の種類から一般的に想定される読者層を考慮してください。
- トーン＆マナー: 「{original_prompt}」に含まれる雰囲気（例：「面白おかしく」「真剣に」「感動的に」など）を最優先してください。
- 主要な要素: 「{original_prompt}」に含まれるキーワード、テーマ、キャラクター、出来事などを中心に据えてください。
- ボリューム: ステップW0のAIがユーザーの指示や内容から提案したボリューム感を尊重してください。特に指定がない場合は、執筆物の種類に応じて一般的な長さを想定してください（例：短編小説なら数千字、ブログ記事なら1500-2000字、レポートなら数ページなど）。
- その他: ユーザーがその執筆物を通して達成したい目的（例：情報提供、娯楽、説得、記録など）を考慮してください。
"""
        finalized_requirements = (
            f"ユーザーの初期リクエスト: 「{original_prompt}」\n\n"
            f"会話全体の主要な目的: 「{initial_user_prompt_for_session}」\n\n"
            f"AIによる要件整理(W0):\n{defined_requirements}\n\n"
            f"{supplemented_info}"
        )
        print(f"ステップW0後 - 最終的な執筆要件 (補足情報込み):\n{finalized_requirements[:500].strip()}...")


        # ステップW1: 構成案・プロット作成
        print("\n執筆モード ステップW1: 構成案・プロット作成中...")
        w1_system_instruction = (
            "あなたは経験豊富な編集者、または作家、シナリオライター、リサーチャーです。\n"
            "提示された詳細な執筆要件を基に、対象となる執筆物の種類に応じた、論理的で分かりやすく、かつ読者/視聴者を引き込むような構成案を作成してください。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            # ... (構成案の指示は変更なし) ...
        )
        w1_user_prompt = (
            f"以下の詳細な執筆要件に基づいて、執筆物の種類に応じた構成案（またはプロット、章立てなど）を作成してください。\n"
            f"--- 確定した執筆要件 ---\n{finalized_requirements}\n--- 執筆要件ここまで ---\n"
            # ... (構成案作成の指示は変更なし) ...
        )
        w1_res = await get_claude_response(
            prompt_text=w1_user_prompt,
            system_instruction=w1_system_instruction,
            model="claude-3-opus-20240229",
            chat_history=list(current_chat_history_for_this_turn), # W0の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w1_res)
        if w1_res.error or not w1_res.response:
            raise ValueError(f"執筆モード ステップW1 (構成案作成) 失敗: {w1_res.error or '応答がありませんでした。'}")
        article_outline = w1_res.response
        print(f"ステップW1 - 作成された構成案/プロット (冒頭):\n{article_outline[:300].strip()}...")

        # ... (以降のステップ W2～W5 も同様に、各AIヘルパー関数呼び出し時に chat_history と initial_user_prompt を渡し、
        #     システムプロンプトや指示プロンプトにこれらの文脈情報を適切に組み込む) ...

        # ステップW2: 初稿（ドラフト）執筆
        print("\n執筆モード ステップW2: 初稿（ドラフト）執筆中...")
        w2_system_role_description = (
            "あなたはプロのライター、小説家、研究者、または脚本家です。\n"
            "与えられた構成案と詳細な執筆要件に基づいて、読者/視聴者にとって魅力的で分かりやすい文章コンテンツの初稿を執筆してください。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            # ... (初稿執筆の指示は変更なし) ...
        )
        w2_user_prompt = (
            f"以下の執筆要件と構成案（またはプロットなど）に基づいて、執筆物の初稿を作成してください。\n"
            f"--- 確定した執筆要件 ---\n{finalized_requirements}\n--- 執筆要件ここまで ---\n\n"
            f"--- 作成された構成案/プロット ---\n{article_outline}\n--- 構成案/プロットここまで ---\n"
            # ... (初稿作成の指示は変更なし) ...
        )
        w2_res = await get_openai_response(
            prompt_text=w2_user_prompt,
            system_role_description=w2_system_role_description,
            model="gpt-4o",
            chat_history=list(current_chat_history_for_this_turn), # W1の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w2_res)
        if w2_res.error or not w2_res.response:
            raise ValueError(f"執筆モード ステップW2 (初稿執筆) 失敗: {w2_res.error or '応答がありませんでした。'}")
        draft_content = w2_res.response
        print(f"ステップW2 - 作成された初稿 (冒頭):\n{draft_content[:300].strip()}...")

        # ステップW3: 内容レビューと改善提案
        print("\n執筆モード ステップW3: 内容レビューと改善提案中...")
        w3_system_instruction = (
            "あなたは経験豊富なプロの編集者です。提示された執筆物の初稿、元々の執筆要件、そして構成案を注意深く読み込み、詳細なレビューと具体的な改善提案を行ってください。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。"
            # ... (レビュー観点は変更なし) ...
        )
        w3_user_prompt = (
            f"以下の執筆物の初稿について、プロの編集者として詳細なレビューと具体的な改善提案をお願いします。\n\n"
            f"--- 元々の執筆要件 ---\n{finalized_requirements}\n--- 執筆要件ここまで ---\n\n"
            f"--- 事前に作成された構成案/プロット ---\n{article_outline}\n--- 構成案/プロットここまで ---\n\n"
            f"--- AIが執筆した初稿 ---\n{draft_content}\n--- 初稿ここまで ---\n\n"
            f"上記のシステム指示に基づき、この初稿をより魅力的で質の高いコンテンツにするための、鋭い指摘と具体的な改善案をリスト形式で複数提示してください。"
        )
        w3_res = await get_claude_response(
            prompt_text=w3_user_prompt,
            system_instruction=w3_system_instruction,
            model="claude-3-opus-20240229",
            chat_history=list(current_chat_history_for_this_turn), # W2の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w3_res)
        if w3_res.error or not w3_res.response:
            raise ValueError(f"執筆モード ステップW3 (レビュー) 失敗: {w3_res.error or '応答がありませんでした。'}")
        review_and_suggestions = w3_res.response
        print(f"ステップW3 - レビューと改善提案 (冒頭):\n{review_and_suggestions[:300].strip()}...")

        # ステップW4: 推敲・リライト
        print("\n執筆モード ステップW4: 推敲・リライト中...")
        w4_preamble_cohere = (
            "あなたはプロの編集者兼ライターです。提示された執筆物の初稿と、それに対する詳細なレビューおよび改善提案を深く理解し、レビューでの指摘事項を的確に反映させてコンテンツ全体を全面的に推敲・リライトしてください。\n"
            f"この会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」でした。\n"
            # ... (推敲指示は変更なし) ...
        )
        w4_user_prompt = (
            f"以下の執筆物の初稿と、それに対するレビューおよび改善提案があります。\n"
            f"これらを基に、初稿を全面的に推敲・リライトし、より質の高いコンテンツ（第2稿）を作成してください。\n\n"
            # ... (W4のユーザープロンプトの残りは変更なし) ...
            f"--- AIが執筆した初稿 ---\n{draft_content}\n--- 初稿ここまで ---\n\n"
            f"--- 上記初稿へのレビューと改善提案 ---\n{review_and_suggestions}\n--- レビューと改善提案ここまで ---\n\n"
            # ...
        )
        w4_res = await get_cohere_response(
            prompt_text=w4_user_prompt,
            preamble=w4_preamble_cohere,
            model="command-r-plus",
            chat_history=list(current_chat_history_for_this_turn), # W3の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w4_res)
        if w4_res.error or not w4_res.response:
            raise ValueError(f"執筆モード ステップW4 (推敲・リライト) 失敗: {w4_res.error or '応答がありませんでした。'}")
        revised_draft_content = w4_res.response
        print(f"ステップW4 - 推敲・リライトされた第2稿 (冒頭):\n{revised_draft_content[:300].strip()}...")

        # ステップW5: 最終校正と仕上げ
        print("\n執筆モード ステップW5: 最終校正と仕上げ中...")
        # 以前はfinalized_requirementsから口調の指定を抽出していたが、
        # 現在はスタイル指定を適用しないため常にNoneとする。
        user_specified_tone = None

        w5_system_instruction_parts = [
            "あなたは最高の編集長兼最終ライターです。与えられた執筆物（第2稿）を、以下の指示に従って完璧な最終版に仕上げてください。",
            "最終出力は小説・物語本文のみとし、章立て案や解説、ストーリー案内、コメントなど本文以外の要素は一切含めないでください。出力が途中で止まった場合は、続きの物語本文のみを書き足してください。"
        ]
        # スタイル指定は現在適用しない
        
        w5_system_instruction_parts.append(
            f"\nこの会話全体の主要な目的は「{initial_user_prompt_for_session}」であり、ユーザーの元々のリクエストは「{original_prompt}」であったことを常に念頭に置いてください。"
            # ... (最終校正の指示の残りは変更なし) ...
        )
        w5_final_system_instruction = "\n\n".join(w5_system_instruction_parts)
        
        w5_user_prompt = (
            f"以下の執筆物（第2稿）を、上記のシステム指示に従って最終校正し、指定された口調（もしあれば）で最高の形に仕上げてください。\n"
            "出力は物語本文のみとし、案内文やコメント、章立て解説を一切含めないでください。続きが必要な場合は本文だけを書き足してください。\n\n"
            f"--- 推敲・リライトされた執筆物（第2稿） ---\n{revised_draft_content}\n--- 第2稿ここまで ---\n\n"
        )
        w5_res = await get_gemini_response(
            prompt_text=w5_user_prompt,
            system_instruction=w5_final_system_instruction,
            model_name="gemini-1.5-pro-latest",
            chat_history=list(current_chat_history_for_this_turn), # W4の結果を含む履歴
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(w5_res)
        if w5_res.error or not w5_res.response:
            raise ValueError(f"執筆モード ステップW5 (最終校正・仕上げ) 失敗: {w5_res.error or '応答がありませんでした。'}")
        final_article_content = w5_res.response
        print(f"ステップW5 - 完成版コンテンツ (冒頭):\n{final_article_content[:300].strip()}...")

        final_step_response = schemas.IndividualAIResponse( # <<< schemas. を追加
            source="Writing Mode (W5 - Final Content)",
            response=final_article_content
        )
        
        response_shell.step7_final_answer_v2_openai = final_step_response
        response_shell.writing_mode_details = steps_executed
        print("--- 執筆特化モード (全ステップ完了) 終了 ---")

    except ValueError as e:
        error_message = f"執筆特化モードの処理中にエラー: {str(e)}"
        print(error_message)
        response_shell.overall_error = error_message
        response_shell.writing_mode_details = steps_executed
        if not response_shell.step7_final_answer_v2_openai: # エラー時に最終応答フィールドが空なら設定
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse( # <<< schemas. を追加
                source="Writing Mode Error Step", 
                error=str(e),
                response=f"申し訳ありません、処理中にエラーが発生しました。\nエラー内容: {str(e)}"
            )
    return response_shell

# --- 超長文執筆モード ---
async def run_ultra_writing_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    desired_char_count: Optional[int] = None
) -> schemas.CollaborativeResponseV2:

    print("\n--- 超長文執筆モード開始 ---")
    steps_executed: List[schemas.IndividualAIResponse] = []

    try:
        outline_res = await get_openai_response(
            prompt_text=f"次の内容で章立てを提案してください:\n{original_prompt}",
            system_role_description="Long Form Outline Generator",
            chat_history=chat_history_for_ai,
            initial_user_prompt=initial_user_prompt_for_session
        )
        steps_executed.append(outline_res)
        if outline_res.error or not outline_res.response:
            raise ValueError("構成案生成に失敗しました")

        chapters = [c.strip() for c in outline_res.response.splitlines() if c.strip()]
        final_text = ""
        for ch in chapters:
            chapter_res = await get_openai_response(
                prompt_text=f"{ch} を詳細に執筆してください。",
                system_role_description="Chapter Writer: 本文のみを出力し、章タイトルや案内文は含めないでください。",
                chat_history=chat_history_for_ai,
                initial_user_prompt=initial_user_prompt_for_session
            )
            steps_executed.append(chapter_res)
            if chapter_res.response:
                final_text += f"\n{chapter_res.response}\n"

        if desired_char_count:
            while len(final_text) < desired_char_count:
                add_res = await get_openai_response(
                    prompt_text=f"以下の文章を続けて詳しく書いてください。残り{desired_char_count - len(final_text)}文字以上必要です。",
                    system_role_description="Expansion Writer: 本文のみを続けて出力してください。コメントや案内は禁止です。",
                    chat_history=chat_history_for_ai,
                    initial_user_prompt=initial_user_prompt_for_session
                )
                steps_executed.append(add_res)
                if not add_res.response:
                    break
                final_text += add_res.response

        response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
            source="Ultra LongWriting Final",
            response=final_text
        )
        response_shell.ultra_writing_mode_details = steps_executed
        print("--- 超長文執筆モード終了 ---")

    except Exception as e:
        error_msg = f"超長文執筆モードの処理中にエラー: {str(e)}"
        print(error_msg)
        response_shell.overall_error = error_msg
        response_shell.ultra_writing_mode_details = steps_executed
        if not response_shell.step7_final_answer_v2_openai:
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="Ultra LongWriting Error",
                error=str(e),
                response=f"申し訳ありません、処理中にエラーが発生しました。\nエラー内容: {str(e)}"
            )

    return response_shell


async def run_fast_chat_mode_flow(
    original_prompt: str,
    response_shell: schemas.CollaborativeResponseV2,
    chat_history_for_ai: List[Dict[str, str]],
    initial_user_prompt_for_session: Optional[str],
    model: str = "gpt-4o"
) -> schemas.CollaborativeResponseV2:
    print("\n--- 高速チャットモード開始 ---")
    full_history = list(chat_history_for_ai)
    full_history.append({"role": "user", "content": original_prompt})
    res = await get_openai_response(
        prompt_text=original_prompt,
        system_role_description="Fast Chat Mode",
        model=model,
        chat_history=full_history,
        initial_user_prompt=initial_user_prompt_for_session,
    )
    response_shell.step7_final_answer_v2_openai = res
    print("--- 高速チャットモード終了 ---")
    return response_shell

    # サーバー起動コマンド (開発時): uvicorn main:app --reload