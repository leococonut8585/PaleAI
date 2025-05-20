# auth_utils.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone as dt_timezone # timezone のインポート名を変更
from typing import Optional
import os
from dotenv import load_dotenv
# from . import schemas # schemas.py を同じ階層からインポートする場合 (後で修正するかも)
# 現状は schemas.py と同じ階層にあるので、以下のように直接インポートできるはず
# もし routers/ フォルダなどを作る場合は、このインポートパスは調整が必要になります。
# import schemas # または具体的にクラス名を指定して from schemas import TokenData など

load_dotenv()

# パスワードハッシュ化の設定 (bcryptを使用)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT設定
SECRET_KEY = os.getenv("SECRET_KEY") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 300))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """平文パスワードとハッシュ化済みパスワードを比較する"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """平文パスワードをハッシュ化する"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """アクセストークンを生成する"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(dt_timezone.utc) + expires_delta
    else:
        expire = datetime.now(dt_timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# トークンをデコードしてユーザーID（この場合はemail）を取得する関数は、
# dependencies.py の get_current_user で実装するので、ここでは不要です。