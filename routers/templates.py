from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List

import database
import models
import schemas
from dependencies import get_current_active_user

router = APIRouter(
    prefix="/templates",
    tags=["PromptTemplates"],
)

@router.post("", response_model=schemas.PromptTemplateResponse, dependencies=[Depends(get_current_active_user)])
async def create_template(
    template_in: schemas.PromptTemplateCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    new_tpl = models.PromptTemplate(
        user_id=current_user.id,
        title=template_in.title,
        content=template_in.content,
        category=template_in.category,
    )
    db.add(new_tpl)
    db.commit()
    db.refresh(new_tpl)
    return new_tpl


@router.get("", response_model=List[schemas.PromptTemplateResponse], dependencies=[Depends(get_current_active_user)])
async def list_templates(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    templates = (
        db.query(models.PromptTemplate)
        .filter(models.PromptTemplate.user_id == current_user.id)
        .order_by(models.PromptTemplate.updated_at.desc())
        .all()
    )
    return templates


@router.put("/{template_id}", response_model=schemas.PromptTemplateResponse, dependencies=[Depends(get_current_active_user)])
async def update_template(
    template_id: int,
    template_in: schemas.PromptTemplateUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    tpl = db.query(models.PromptTemplate).filter(
        models.PromptTemplate.id == template_id,
        models.PromptTemplate.user_id == current_user.id
    ).first()
    if not tpl:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    if template_in.title is not None:
        tpl.title = template_in.title
    if template_in.content is not None:
        tpl.content = template_in.content
    if template_in.category is not None:
        tpl.category = template_in.category
    tpl.updated_at = func.now()
    db.add(tpl)
    db.commit()
    db.refresh(tpl)
    return tpl


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_active_user)])
async def delete_template(
    template_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    tpl = db.query(models.PromptTemplate).filter(
        models.PromptTemplate.id == template_id,
        models.PromptTemplate.user_id == current_user.id
    ).first()
    if not tpl:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    db.delete(tpl)
    db.commit()
    return
