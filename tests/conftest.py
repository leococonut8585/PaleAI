import os
import subprocess
import pytest

# 環境変数は database モジュールを読み込む前に設定する
os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from database import engine


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    print("\n--- setup_database: 開始 (テスト関数ごと) ---")
    db_path = "test.db"
    if os.path.exists(db_path):
        print(f"  {db_path} が存在するため、engine.dispose() と os.remove() を実行します。")
        engine.dispose()
        os.remove(db_path)
    else:
        print(f"  {db_path} は存在しません。新規作成の準備をします。")

    print("  alembic upgrade head を実行します...")
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    print("  alembic upgrade head 完了。yieldでテストを実行します。")

    yield

    print("\n--- setup_database: テスト関数終了後のクリーンアップ開始 ---")
    if os.path.exists(db_path):
        print(f"  {db_path} が存在するため、engine.dispose() と os.remove() を実行します。 (クリーンアップ)")
        engine.dispose()
        os.remove(db_path)
    else:
        print(f"  {db_path} はクリーンアップ時には既に存在しませんでした。")
    print("--- setup_database: クリーンアップ完了 ---")
