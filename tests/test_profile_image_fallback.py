import shutil
from pathlib import Path
import pytest

import services.profile_image as pi


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def get(self, url):
        class Resp:
            def __init__(self, content):
                self.content = content
        return Resp(Path("static/pic/Default.png").read_bytes())


@pytest.mark.asyncio
async def test_dalle_error_triggers_sdxl(monkeypatch):
    # ensure clean dir
    out = Path("static/profile")
    if out.exists():
        shutil.rmtree(out)

    async def raise_error(*args, **kwargs):
        raise Exception("dalle fail")

    monkeypatch.setattr(pi.openai_client.images, "generate", raise_error)
    monkeypatch.setattr(pi.sd_client, "run", lambda *a, **k: "dummy")
    monkeypatch.setattr(pi, "httpx", type("T", (), {"AsyncClient": DummyAsyncClient}))

    await pi.generate_and_save("#000000", "#ffffff", "male", 1)

    img_path = Path("static/profile/1.png")
    assert img_path.exists()
    assert img_path.stat().st_size > 1024
