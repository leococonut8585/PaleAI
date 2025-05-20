import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from main import app
import os

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/folders")
    assert resp.status_code in (401, 403)


def test_search_requires_auth():
    resp = client.get("/chat/search?q=test")
    assert resp.status_code in (401, 403)


def test_upload_requires_auth():
    resp = client.post("/upload", files={"file": ("test.txt", b"hello")})
    assert resp.status_code in (401, 403)


def test_chat_upload_requires_auth():
    resp = client.post("/chat/upload/1", files={"file": ("test.txt", b"hello")})
    assert resp.status_code in (401, 403)


def test_memory_requires_auth():
    resp = client.get("/memory")
    assert resp.status_code in (401, 403)


def test_templates_requires_auth():
    resp = client.get("/templates")
    assert resp.status_code in (401, 403)



def test_register_login_and_me():
    reg = client.post(
        "/auth/register",
        json={
            "email": "u@example.com",
            "username": "user1",
            "password": "pass1234",
            "gender": "未回答",
            "color1": "#000000",
            "color2": "#ffffff",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    assert data["username"] == "user1"
    img_path = "." + data["profile_image_url"]
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1024

    token_res = client.post(
        "/auth/login", data={"username": "user1", "password": "pass1234"}
    )
    assert token_res.status_code == 200
    token = token_res.json()["access_token"]

    me_res = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    data = me_res.json()
    assert data["username"] == "user1"
    assert data["gender"] == "未回答"

    dup = client.post(
        "/auth/register",
        json={
            "email": "u2@example.com",
            "username": "user1",
            "password": "x",
            "gender": "未回答",
            "color1": "#000000",
            "color2": "#ffffff",
        },
    )
    assert dup.status_code == 400

def test_profile_image_creation_and_update():
    reg = client.post(
        "/auth/register",
        json={
            "email": "u2@example.com",
            "username": "user2",
            "password": "pass1234",
            "gender": "男性",
            "color1": "#112233",
            "color2": "#445566",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    path = "." + data["profile_image_url"]
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1024

    token_res = client.post(
        "/auth/login", data={"username": "user2", "password": "pass1234"}
    )
    token = token_res.json()["access_token"]
    upd = client.put(
        "/users/me",
        json={"color1": "#223344"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert upd.status_code == 200
    new_path = "." + upd.json()["profile_image_url"]
    assert new_path != path
    assert os.path.exists(new_path)
    assert os.path.getsize(new_path) > 1024
