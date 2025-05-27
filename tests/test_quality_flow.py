import pytest
from datetime import datetime
from starlette.requests import Request

import ai_processing_flows as flows
import schemas

class DummyPerplexity:
    def __init__(self):
        self.model = None
        self.queries = []
    def query(self, prompt):
        self.queries.append(prompt)
        return "x" * 3100

class DummyClaudeMessages:
    def __init__(self):
        self.calls = []
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Res", (), {"content": [type("Block", (), {"text": "summary"})()]})

class DummyClaudeClient:
    def __init__(self):
        self.messages = DummyClaudeMessages()

@pytest.mark.asyncio
async def test_quality_chat_mode_flow():
    app = type("App", (), {"state": type("State", (), {})()})()
    app.state.perplexity_sync_client = DummyPerplexity()
    app.state.anthropic_client = DummyClaudeClient()

    req = Request({"type": "http", "app": app})
    shell = schemas.CollaborativeResponseV2(prompt="q")

    res = await flows.run_quality_chat_mode_flow(
        original_prompt="テスト",
        response_shell=shell,
        chat_history_for_ai=[],
        initial_user_prompt_for_session=None,
        user_memories=None,
        request=req,
    )

    queried = app.state.perplexity_sync_client.queries[0]
    assert datetime.utcnow().strftime("%Y-%m-%d") in queried
    assert "1000文字以上" in queried
    assert "重複" in queried

    claude_call = app.state.anthropic_client.messages.calls[0]
    user_msg = [m for m in claude_call["messages"] if m["role"] == "user"][0]
    assert "3000文字以上" in user_msg["content"]

    assert res.step4_comprehensive_answer_perplexity.response.startswith("x")
    assert res.step7_final_answer_v2_openai.response == "summary"
