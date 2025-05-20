from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List

import database
import models
import schemas
from dependencies import get_current_active_user

router = APIRouter(
    prefix="/folders",
    tags=["Folders"],
)

@router.post("", response_model=schemas.FolderResponse)
async def create_folder(
    folder_in: schemas.FolderCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    max_pos = db.query(func.max(models.Folder.position)).filter(models.Folder.user_id == current_user.id).scalar()
    next_pos = (max_pos + 1) if max_pos is not None else 0
    new_folder = models.Folder(name=folder_in.name, user_id=current_user.id, position=next_pos)
    db.add(new_folder)
    db.commit()
    db.refresh(new_folder)
    return new_folder

@router.get("", response_model=List[schemas.FolderResponse])
async def list_folders(
    search: str | None = None,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    query = db.query(models.Folder).filter(models.Folder.user_id == current_user.id)
    if search:
        query = query.filter(models.Folder.name.ilike(f"%{search}%"))
    folders = query.order_by(models.Folder.position.asc().nullsfirst(), models.Folder.created_at.asc()).all()
    return folders

@router.put("/{folder_id}", response_model=schemas.FolderResponse)
async def update_folder(
    folder_id: int,
    folder_in: schemas.FolderUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    folder = db.query(models.Folder).filter(
        models.Folder.id == folder_id, models.Folder.user_id == current_user.id
    ).first()
    if not folder:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found")
    if folder_in.name is not None:
        folder.name = folder_in.name
    folder.updated_at = func.now()
    db.add(folder)
    db.commit()
    db.refresh(folder)
    return folder

@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    folder = db.query(models.Folder).filter(
        models.Folder.id == folder_id, models.Folder.user_id == current_user.id
    ).first()
    if not folder:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found")
    db.delete(folder)
    db.commit()
    return

@router.put("/{folder_id}/sessions/{session_id}", response_model=schemas.ChatSessionResponse)
async def move_chat_session(
    folder_id: int,
    session_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    if folder_id == 0:
        session.folder_id = None
    else:
        folder = db.query(models.Folder).filter(
            models.Folder.id == folder_id, models.Folder.user_id == current_user.id
        ).first()
        if not folder:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found")
        session.folder_id = folder.id
    session.updated_at = func.now()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.patch("/reorder", response_model=List[schemas.FolderResponse])
async def reorder_folders(
    ordered_ids: List[int],
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    folders = db.query(models.Folder).filter(
        models.Folder.user_id == current_user.id,
        models.Folder.id.in_(ordered_ids)
    ).all()
    id_to_folder = {f.id: f for f in folders}
    for position, fid in enumerate(ordered_ids):
        folder = id_to_folder.get(fid)
        if folder:
            folder.position = position
            db.add(folder)
    db.commit()
    return list(id_to_folder.values())
