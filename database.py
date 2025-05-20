# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv() # .envファイルから環境変数を読み込む

# .envファイルからDATABASE_URLを読み込む。なければデフォルトでSQLiteを使用。
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pale_ai.db")

# データベースエンジンを作成
# SQLiteの場合、FastAPIの複数スレッドからのアクセスに対応するために connect_args={"check_same_thread": False} が必要
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

# データベースセッションを作成するためのSessionLocalクラス
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemyモデルを定義するためのベースクラス
Base = declarative_base()

# APIエンドポイントでデータベースセッションを取得するための依存関係関数
def get_db():
    db = SessionLocal()
    try:
        yield db  # セッションをエンドポイント関数に提供
    finally:
        db.close() # 処理が終わったらセッションを閉じる