from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import models
import schemas
from database import get_db
from dependencies import get_current_active_user

router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    dependencies=[Depends(get_current_active_user)],
)

MAX_MEMORIES_PER_USER = 100

@router.get("", response_model=List[schemas.UserMemoryResponse])
async def get_all_user_memories(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """現在のユーザーの全てのメモリを取得します。"""
    memories = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.user_id == current_user.id)
        .order_by(models.UserMemory.priority.desc(), models.UserMemory.updated_at.desc())
        .all()
    )
    return memories

@router.post("", response_model=schemas.UserMemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_new_memory(
    memory_data: schemas.UserMemoryCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """新しいメモリを作成します。ユーザーあたりのメモリ上限は100個です。"""
    current_count = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.user_id == current_user.id)
        .count()
    )
    if current_count >= MAX_MEMORIES_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"メモリの最大数 ({MAX_MEMORIES_PER_USER}個) に達しています。",
        )

    new_memory = models.UserMemory(**memory_data.model_dump(), user_id=current_user.id)
    db.add(new_memory)
    db.commit()
    db.refresh(new_memory)
    return new_memory

@router.get("/{memory_id}", response_model=schemas.UserMemoryResponse)
async def get_specific_memory(
    memory_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """特定のIDのメモリを取得します。"""
    memory = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.id == memory_id, models.UserMemory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="メモリが見つかりません。")
    return memory

@router.put("/{memory_id}", response_model=schemas.UserMemoryResponse)
async def update_existing_memory(
    memory_id: int,
    memory_update_data: schemas.UserMemoryUpdate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """既存のメモリを更新します。"""
    memory = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.id == memory_id, models.UserMemory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="更新対象のメモリが見つかりません。")

    update_data = memory_update_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(memory, key, value)

    db.commit()
    db.refresh(memory)
    return memory

@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_specific_memory(
    memory_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """特定のIDのメモリを削除します。"""
    memory = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.id == memory_id, models.UserMemory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="削除対象のメモリが見つかりません。")

    db.delete(memory)
    db.commit()
    return None
