# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
import logging

# routers/users.py や routers/auth.py から見て、
# これらのファイルがプロジェクトルートにあることを想定したインポート
import auth_utils
import models
import schemas
import database

# tokenUrl は、トークンを取得するためのログインエンドポイントのパスを指定します。
# routers/auth.py で @router.post("/login", ...) と定義し、
# main.py で auth.router の prefix が "/auth" なので、フルパスは "/auth/login" になります。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

logger = logging.getLogger(__name__)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)) -> models.User:
    logger.debug(
        "DEBUG: get_current_user called with token (first 10 chars): %s",
        token[:10],
    )

    # ↓↓↓ ここを修正 ↓↓↓
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="有効な認証情報を提供できませんでした",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # ↑↑↑ 修正ここまで ↑↑↑
    try:
        payload = jwt.decode(token, auth_utils.SECRET_KEY, algorithms=[auth_utils.ALGORITHM])
        email: str = payload.get("sub")
        logger.debug("DEBUG: Token decoded. Email from payload: %s", email)
        if email is None:
            logger.debug("DEBUG: Email is None in payload.")
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError as e:
        logger.debug("DEBUG: JWTError during token decoding: %s", e)
        raise credentials_exception

    user = db.query(models.User).filter(models.User.email == token_data.email).first()
    if user is None:
        logger.debug("DEBUG: User not found in DB for email: %s", token_data.email)
        raise credentials_exception
    logger.debug(
        "DEBUG: User found in DB: ID=%s, Email=%s, IsActive=%s",
        user.id,
        user.email,
        user.is_active,
    )
    if not user.is_active:
        logger.debug("DEBUG: User %s is not active.", user.email)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非アクティブなアカウントです。")
    return user

async def get_current_active_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    """
    get_current_user を使ってユーザーを取得し、アクティブであるか確認する。
    保護されたエンドポイントで、現在アクティブなユーザーのみを許可する場合に使用する。
    """
    if not current_user.is_active: 
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非アクティブなアカウントです。")
    return current_user
