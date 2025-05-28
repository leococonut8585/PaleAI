# routers/users.py
from fastapi import APIRouter, Depends, Request # Added Request
from sqlalchemy.orm import Session
from typing import Any # Added Any

# プロジェクトルートにあるモジュールを直接インポート
import schemas
import models
import dependencies  # dependencies.py もプロジェクトルートにある想定
import database
from . import auth

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(dependencies.get_current_active_user)]
)

@router.get("/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(dependencies.get_current_active_user)):
    """
    現在認証されているユーザーの情報を取得します。
    """
    return current_user


@router.put("/me", response_model=schemas.User)
async def update_users_me(
    update_data: schemas.UserUpdate, # Changed 'update' to 'update_data'
    request: Request, # Added request
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_active_user),
) -> models.User: # Explicitly added -> models.User return type
    """Update current user's gender or colors and regenerate profile image."""
    if update_data.gender is not None:
        current_user.gender = update_data.gender
    if update_data.color1 is not None:
        current_user.color1 = update_data.color1
    if update_data.color2 is not None:
        current_user.color2 = update_data.color2
    db.commit()
    openai_client_instance = request.app.state.openai_client # Get client instance
    await auth.create_profile_image(current_user, db, openai_api_client=openai_client_instance) # Pass client
    db.refresh(current_user)
    return current_user
