# routers/chat.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy.sql import func # ★★★ func をインポート ★★★
from typing import List

from file_processors import stage0_process

import database
import models
import schemas
from dependencies import get_current_active_user

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(get_current_active_user)]
)

# --- チャットセッション関連 ---

@router.post("/sessions", response_model=schemas.ChatSessionResponse)
async def create_chat_session(
    chat_session_in: schemas.ChatSessionCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    新しいチャットセッションを開始します。
    タイトルはクライアントから指定されるか、AI処理エンドポイントで自動生成されます。
    """
    title_to_set = chat_session_in.title
    if not title_to_set: # フロントエンドがタイトル未指定でAPIを叩くことも考慮
        title_to_set = "新しいチャット" # デフォルトの仮タイトル

    new_session = models.ChatSession(
        user_id=current_user.id,
        title=title_to_set,
        starred=chat_session_in.starred if hasattr(chat_session_in, 'starred') else False,
        tags=chat_session_in.tags if hasattr(chat_session_in, 'tags') else None,
        mode='chat',
        is_complete=True,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

@router.get("/sessions", response_model=List[schemas.ChatSessionResponse])
async def get_user_chat_sessions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    sessions = db.query(models.ChatSession)\
                 .filter(models.ChatSession.user_id == current_user.id)\
                 .filter(models.ChatSession.is_complete == True)\
                 .order_by(models.ChatSession.updated_at.desc())\
                 .offset(skip)\
                 .limit(limit)\
                 .all()
    return sessions

@router.get("/sessions/{session_id}/messages", response_model=List[schemas.ChatMessageResponse]) # ★★★ エンドポイントを修正 ★★★
async def get_chat_session_messages( # ★★★ 関数名を修正 ★★★
    session_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    特定のチャットセッションのメッセージ履歴を取得します。
    main.py にも同様のエンドポイントがありましたが、認証や責務的にこちらにある方が自然です。
    main.py側のものは削除またはコメントアウトを検討してください。
    """
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="チャットセッションが見つかりません")

    messages = db.query(models.ChatMessage)\
                 .filter(models.ChatMessage.chat_session_id == session_id)\
                 .order_by(models.ChatMessage.created_at.asc())\
                 .all()
    return messages

@router.put("/sessions/{session_id}/title", response_model=schemas.ChatSessionResponse) # ★★★ 追加 ★★★
async def update_chat_session_title(
    session_id: int,
    title_update: schemas.ChatSessionTitleUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    chat_session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id
    ).first()

    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found or access denied"
        )

    chat_session.title = title_update.title
    chat_session.updated_at = func.now()
    db.add(chat_session) # SQLAlchemyが変更を追跡するので、addは必須ではないが明示的に
    db.commit()
    db.refresh(chat_session)
    return chat_session

@router.put("/sessions/{session_id}/star", response_model=schemas.ChatSessionResponse)
async def update_chat_session_star(
    session_id: int,
    star_update: schemas.ChatSessionStarUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    chat_session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id
    ).first()
    if not chat_session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied")
    chat_session.starred = star_update.starred
    chat_session.updated_at = func.now()
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session

@router.put("/sessions/{session_id}/tags", response_model=schemas.ChatSessionResponse)
async def update_chat_session_tags(
    session_id: int,
    tags_update: schemas.ChatSessionTagsUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    chat_session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id
    ).first()
    if not chat_session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied")
    chat_session.tags = tags_update.tags
    chat_session.updated_at = func.now()
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT) # ★★★ 追加 ★★★
async def delete_chat_session(
    session_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    chat_session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id
    ).first()

    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found or access denied"
        )

    db.delete(chat_session)
    db.commit()
    return


# --- チャットメッセージ関連 (既存のものは一旦そのまま) ---
# メッセージ追加は主に /collaborative_answer_v2 でAI応答と共に保存されるため、
# ユーザーメッセージだけを保存するこのエンドポイントの扱いは要検討。
# 現状のフロントエンドでは /collaborative_answer_v2 を使っている。

@router.post("/sessions/{session_id}/messages", response_model=schemas.ChatMessageResponse, include_in_schema=False) # 一旦スキーマ非表示
async def add_message_to_session(
    session_id: int,
    message_in: schemas.ChatMessageCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    session = db.query(models.ChatSession)\
                .filter(models.ChatSession.id == session_id, models.ChatSession.user_id == current_user.id)\
                .first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="チャットセッションが見つかりません")

    new_message = models.ChatMessage(
        chat_session_id=session.id,
        user_id=current_user.id, # ★★★ ユーザーIDも保存 ★★★
        role=message_in.role,
        content=message_in.content,
        ai_model=message_in.ai_model
    )
    db.add(new_message)
    session.updated_at = func.now()
    db.add(session) # SQLAlchemyが変更を追跡するので、addは必須ではないが明示的に
    db.commit()
    db.refresh(new_message)
    return new_message

@router.post("/sessions/{session_id}/clone", response_model=schemas.ChatSessionResponse)
async def clone_chat_session(
    session_id: int,
    clone_data: schemas.ChatSessionClone,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    original = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id,
    ).first()
    if not original:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    new_session = models.ChatSession(
        user_id=current_user.id,
        title=clone_data.title or f"{original.title} (copy)",
        folder_id=clone_data.folder_id,
        mode=original.mode,
        is_complete=original.is_complete,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    messages = db.query(models.ChatMessage).filter(models.ChatMessage.chat_session_id == original.id).order_by(models.ChatMessage.created_at.asc()).all()
    for msg in messages:
        db.add(models.ChatMessage(
            chat_session_id=new_session.id,
            user_id=msg.user_id,
            role=msg.role,
            content=msg.content,
            ai_model=msg.ai_model,
        ))
    new_session.updated_at = func.now()
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

@router.get("/search", response_model=List[schemas.MessageSearchResult])
async def search_messages(
    q: str,
    limit: int = 20,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    if not q:
        return []
    pattern = f"%{q}%"
    results = (
        db.query(models.ChatMessage)
        .join(models.ChatSession, models.ChatMessage.chat_session_id == models.ChatSession.id)
        .filter(models.ChatSession.user_id == current_user.id)
        .filter(models.ChatMessage.content.ilike(pattern))
        .order_by(models.ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        schemas.MessageSearchResult(
            session_id=r.chat_session_id,
            message_id=r.id,
            content=r.content,
        )
        for r in results
    ]


@router.post("/upload/{session_id}", response_model=schemas.ChatMessageResponse)
async def upload_file_to_chat(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied")

    content = await file.read()
    result = await stage0_process(file.filename, content)

    new_message = models.ChatMessage(
        chat_session_id=session.id,
        user_id=None,
        role="system",
        content=result["summary"],
        ai_model="stage0",
    )
    session.updated_at = func.now()
    db.add(new_message)
    db.add(session)
    db.commit()
    db.refresh(new_message)
    return new_message
