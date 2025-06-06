# routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import timedelta
import logging
from typing import Any # Added
from fastapi import Request # Added

# プロジェクトルートにあるモジュールを直接インポート
import database
import models
import schemas
import auth_utils
from services.profile_image import generate_and_save
import uuid
import os

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

logger = logging.getLogger(__name__)

async def create_profile_image(user: models.User, db: Session, openai_api_client: Any) -> None:
    """Generate and save the user's profile image."""
    unique_name = f"{user.id}_{uuid.uuid4().hex}.png"
    path = f"/static/profile/{unique_name}"
    old_path = user.profile_image_url
    user.profile_image_url = path
    db.commit()
    try:
        # Pass the openai_api_client to generate_and_save
        await generate_and_save(
            user.color1, user.color2, user.gender, user.id, openai_api_client, file_name=unique_name
        )
        if old_path:
            try:
                os.remove("." + old_path)
            except OSError:
                pass
    except Exception as e:
        logger.error("Profile image generation error: %s", e)

@router.post("/register", response_model=schemas.User)
async def register_user(
    user_in: schemas.UserCreate,
    request: Request, 
    bg: BackgroundTasks,
    db: Session = Depends(database.get_db),
):
    """
    新しいユーザーを登録します。
    - **email**: ユーザーのメールアドレス (必須、一意)
    - **password**: ユーザーのパスワード (必須)
    """
    if db.query(models.User).filter(models.User.email == user_in.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="このメールアドレスは既に使用されています。")
    if db.query(models.User).filter(models.User.username == user_in.username).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="このユーザー名は既に使用されています。")

    hashed_password = auth_utils.get_password_hash(user_in.password)
    new_user = models.User(
        email=user_in.email,
        username=user_in.username,
        gender=user_in.gender,
        color1=user_in.color1,
        color2=user_in.color2,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    new_user.profile_image_url = f"/static/profile/{new_user.id}.png"
    db.commit()
    openai_client_instance = request.app.state.openai_client # Get client instance
    bg.add_task(
        generate_and_save, new_user.color1, new_user.color2, new_user.gender, new_user.id, openai_client_instance, None # Pass client and None for file_name
    )
    return new_user

@router.post("/login", response_model=schemas.Token)
async def login_for_access_token(
    request: Request, 
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(database.get_db)
):
    """
    ユーザーを認証し、アクセストークンを発行します。
    - **username**: メールアドレスまたはユーザー名 (フォームデータとして)
    - **password**: ユーザーのパスワード (フォームデータとして)
    """
    user = db.query(models.User).filter(
        (models.User.email == form_data.username) | (models.User.username == form_data.username)
    ).first()
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="メールアドレスまたはパスワードが正しくありません。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非アクティブなアカウントです。")

    if not user.profile_image_url:
        openai_client_instance = request.app.state.openai_client # Get client instance
        await create_profile_image(user, db, openai_api_client=openai_client_instance) # Pass client

    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
