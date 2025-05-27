import pytest
from datetime import datetime
from starlette.requests import Request

import main
from main import schemas

class DummyPerplexity:
    def __init__(self, responses=None):
        self.model = None
        self.queries = []
        self.responses = responses or ["x" * 1100]
        self.call_count = 0
    def query(self, prompt):
        self.queries.append(prompt)
        resp = self.responses[min(self.call_count, len(self.responses)-1)]
        self.call_count += 1
        return resp

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
def test_balance_mode_perplexity_year():
    app = type("App", (), {"state": type("State", (), {})()})()
    app.state.perplexity_sync_client = DummyPerplexity([f"info {datetime.utcnow().year}"])
    app.state.anthropic_client = DummyClaudeClient()

    req = Request({"type": "http", "app": app})
    shell = schemas.CollaborativeResponseV2(prompt="q")

    res = await main.run_balance_mode_flow(
        original_prompt="最新のAIニュースは？",
        response_shell=shell,
        chat_history_for_ai=[],
        initial_user_prompt_for_session="AIについて",
        request=req,
        user_memories=None,
    )

    query = app.state.perplexity_sync_client.queries[0]
    assert datetime.utcnow().strftime("%Y-%m-%d") in query
    assert f"{datetime.utcnow().year}" in res.step4_comprehensive_answer_perplexity.response
    assert res.step7_final_answer_v2_openai.response == "summary"

@pytest.mark.asyncio
def test_balance_mode_perplexity_retry():
    short = "short"
    long_resp = "x" * 1200
    app = type("App", (), {"state": type("State", (), {})()})()
    app.state.perplexity_sync_client = DummyPerplexity([short, long_resp])
    app.state.anthropic_client = DummyClaudeClient()

    req = Request({"type": "http", "app": app})
    shell = schemas.CollaborativeResponseV2(prompt="q")

    res = await main.run_balance_mode_flow(
        original_prompt="古いAIニュースは？",
        response_shell=shell,
        chat_history_for_ai=[],
        initial_user_prompt_for_session="AIについて",
        request=req,
        user_memories=None,
    )

    assert app.state.perplexity_sync_client.call_count == 2
    assert len(res.step4_comprehensive_answer_perplexity.response) >= len(long_resp)
    assert res.step7_final_answer_v2_openai.response == "summary"
