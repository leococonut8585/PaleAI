from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Request
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import uuid
import os
import logging

import models
import schemas
from dependencies import get_current_active_user
from database import get_db

from ai_processing_flows import (
    run_quality_chat_mode_flow,
    run_ultra_search_flow,
)
from file_processors import stage0_process_file

try:
    from utils.memory_retriever import get_relevant_memories_for_prompt
except Exception:  # pragma: no cover - placeholder when module missing

    async def get_relevant_memories_for_prompt(*args, **kwargs):
        return []


UPLOAD_DIR = "uploaded_files_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

logger = logging.getLogger(__name__)


@router.post("")
async def root_upload_placeholder(
    current_user: models.User = Depends(get_current_active_user),
):
    """Placeholder root endpoint requiring authentication."""
    return {"detail": "not implemented"}


@router.post("/process_file_and_chat/", response_model=schemas.CollaborativeResponseV2)
async def process_file_and_chat_endpoint(
    request: Request,
    file: UploadFile = File(...),
    current_chat_session_id: Optional[str] = Form(None),
    current_ai_mode: str = Form(...),
    original_prompt: str = Form(""),
    initial_user_prompt_for_session: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    request_id = str(uuid.uuid4())

    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル読み込みエラー: {str(e)}")
    finally:
        await file.close()

    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイル名がありません。")

    stage0_result = await stage0_process_file(request, file.filename, file_content)

    if stage0_result["error"]:
        raise HTTPException(
            status_code=stage0_result["status_code"], detail=stage0_result["error"]
        )

    processed_file_text = stage0_result["processed_content"]

    response_shell = schemas.CollaborativeResponseV2(
        request_id=request_id,
        mode=current_ai_mode,
        final_user_prompt=original_prompt,
        uploaded_file_info={
            "filename": stage0_result.get("original_filename", file.filename),
            "content_type": stage0_result.get("content_type", file.content_type),
            "size_bytes": stage0_result.get("size_bytes", len(file_content)),
            "processing_ai": stage0_result.get("processing_ai", "N/A"),
        },
        step0_file_processing_result=schemas.IndividualAIResponse(
            source=stage0_result.get("processing_ai", "FileProcessor"),
            prompt_text="N/A (File Content used as input)",
            response=(
                processed_file_text
                if processed_file_text
                else "[ファイル内容の処理結果なし]"
            ),
            error=None,
        ),
    )

    user_prompt_for_ai = original_prompt
    if not user_prompt_for_ai.strip() and processed_file_text:
        user_prompt_for_ai = f"このファイル「{file.filename}」の内容について説明、または要約してください。"

    prompt_with_file_context = (
        f"ユーザーはファイル「{file.filename}」をアップロードしました。\n"
        f"このファイルは「{stage0_result.get('processing_ai', '不明な方法')}」によって処理され、以下の内容が抽出されました（または記述されました）。\n"
        f"--- ファイル内容ここから ---\n"
        f"{processed_file_text if processed_file_text else '[ファイルからの抽出内容なし]'}\n"
        f"--- ファイル内容ここまで ---\n\n"
        f"上記のファイル内容を踏まえ、以下のユーザーの指示に答えてください:\n"
        f"{user_prompt_for_ai}"
    )
    response_shell.final_user_prompt = prompt_with_file_context

    chat_history_for_ai: List[Dict[str, str]] = []
    if current_chat_session_id:
        pass

    user_memories_list: List[schemas.UserMemoryResponse] = []
    if current_user and getattr(current_user, "memories", None):
        pass

    try:
        if current_ai_mode == "quality":
            response_shell = await run_quality_chat_mode_flow(
                original_prompt=prompt_with_file_context,
                response_shell=response_shell,
                chat_history_for_ai=chat_history_for_ai,
                initial_user_prompt_for_session=initial_user_prompt_for_session,
                user_memories=user_memories_list,
                request=request,
            )
        elif current_ai_mode == "ultrasearch":
            response_shell = await run_ultra_search_flow(
                original_prompt=prompt_with_file_context,
                response_shell=response_shell,
                chat_history_for_ai=chat_history_for_ai,
                user_memories=user_memories_list,
                request=request,
            )
        else:
            error_msg = f"指定されたAIモード '{current_ai_mode}' はファイルアップロード処理に対応していません。"
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="System",
                response=f"エラー: {error_msg}",
                error=error_msg,
            )
    except HTTPException:
        raise
    except Exception as e:
        error_message = (
            f"AI処理フロー ({current_ai_mode}) でエラーが発生しました: {str(e)}"
        )
        logger.error("[Error] %s", error_message)
        response_shell.overall_error = error_message
        if (
            not response_shell.step7_final_answer_v2_openai
            or not response_shell.step7_final_answer_v2_openai.response
        ):
            response_shell.step7_final_answer_v2_openai = schemas.IndividualAIResponse(
                source="System Error",
                response="申し訳ありません、AIの応答生成中に内部エラーが発生しました。",
                error=str(e),
            )

    return response_shell
