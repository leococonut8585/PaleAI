# routers/users.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

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
    update: schemas.UserUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_active_user),
):
    """Update current user's gender or colors and regenerate profile image."""
    if update.gender is not None:
        current_user.gender = update.gender
    if update.color1 is not None:
        current_user.color1 = update.color1
    if update.color2 is not None:
        current_user.color2 = update.color2
    db.commit()
    await auth.create_profile_image(current_user, db)
    db.refresh(current_user)
    return current_user
