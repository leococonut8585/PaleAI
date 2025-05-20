from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List

import database
import models
import schemas
from dependencies import get_current_active_user

router = APIRouter(
    prefix="/memory",
    tags=["Memory"],
)

@router.post("", response_model=schemas.UserMemoryResponse, dependencies=[Depends(get_current_active_user)])
async def create_memory(
    memory_in: schemas.UserMemoryCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    new_memory = models.UserMemory(
        user_id=current_user.id,
        title=memory_in.title,
        content=memory_in.content,
        priority=memory_in.priority if memory_in.priority is not None else 0,
    )
    db.add(new_memory)
    db.commit()
    db.refresh(new_memory)
    return new_memory

@router.get("", response_model=List[schemas.UserMemoryResponse], dependencies=[Depends(get_current_active_user)])
async def list_memory(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    memories = (
        db.query(models.UserMemory)
        .filter(models.UserMemory.user_id == current_user.id)
        .order_by(models.UserMemory.updated_at.desc())
        .all()
    )
    return memories

@router.put("/{memory_id}", response_model=schemas.UserMemoryResponse, dependencies=[Depends(get_current_active_user)])
async def update_memory(
    memory_id: int,
    memory_in: schemas.UserMemoryUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    memory = db.query(models.UserMemory).filter(
        models.UserMemory.id == memory_id,
        models.UserMemory.user_id == current_user.id
    ).first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    if memory_in.title is not None:
        memory.title = memory_in.title
    if memory_in.content is not None:
        memory.content = memory_in.content
    if memory_in.priority is not None:
        memory.priority = memory_in.priority
    memory.updated_at = func.now()
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory

@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_active_user)])
async def delete_memory(
    memory_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    memory = db.query(models.UserMemory).filter(
        models.UserMemory.id == memory_id,
        models.UserMemory.user_id == current_user.id
    ).first()
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    db.delete(memory)
    db.commit()
    return
