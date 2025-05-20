import os
import subprocess
import pytest

os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    db_path = "test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    yield
    if os.path.exists(db_path):
        os.remove(db_path)
