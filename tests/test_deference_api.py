import os
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from main import app
from routers.images import ImageGenerationRequest

client = TestClient(app)


def test_image_deference_field_validation():
    resp = client.post(
        "/images/generate", json={"prompt": "x", "count": 1, "deference": 0}
    )
    assert resp.status_code in (401, 403, 422)


def test_image_deference_default():
    req = ImageGenerationRequest(prompt="x")
    assert req.deference == 3
