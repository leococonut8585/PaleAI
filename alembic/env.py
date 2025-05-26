from logging.config import fileConfig
import sys
import os

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# --- パス設定 ---
# カレントディレクトリにあるmainアプリケーションのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モデルとBase読み込み ---
from models import Base  # ← あなたのmodels.pyに定義されたBase

# Alembic設定の読み込み
config = context.config

# Logging設定
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 環境変数 DATABASE_URL が設定されていれば Alembic の設定を上書きする
db_url = os.getenv("DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# マイグレーション対象のメタデータを設定
target_metadata = Base.metadata

# --- オフラインモードの処理 ---
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# --- オンラインモードの処理 ---
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

# 実行モードに応じて分岐
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
