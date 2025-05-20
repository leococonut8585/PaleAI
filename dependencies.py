# dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

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

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)) -> models.User:
    print(f"DEBUG: get_current_user called with token (first 10 chars): {token[:10]}...")

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
        print(f"DEBUG: Token decoded. Email from payload: {email}") # デコードされたメール確認
        if email is None:
            print("DEBUG: Email is None in payload.")
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError as e:
        print(f"DEBUG: JWTError during token decoding: {e}") # JWTエラー詳細
        raise credentials_exception

    user = db.query(models.User).filter(models.User.email == token_data.email).first()
    if user is None:
        print(f"DEBUG: User not found in DB for email: {token_data.email}") # ユーザーが見つからない場合
        raise credentials_exception
    print(f"DEBUG: User found in DB: ID={user.id}, Email={user.email}, IsActive={user.is_active}") # ユーザー情報確認
    if not user.is_active:
        print(f"DEBUG: User {user.email} is not active.")
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
