import os
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from main import app

client = TestClient(app)


def test_video_requires_auth():
    resp = client.post("/video/generate", json={"prompt": "test"})
    assert resp.status_code in (401, 403)
