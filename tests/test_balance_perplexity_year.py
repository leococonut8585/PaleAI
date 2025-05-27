import os
import pytest
from datetime import datetime
from starlette.requests import Request

import main
from main import schemas


@pytest.mark.asyncio
async def test_balance_mode_perplexity_year(monkeypatch):
    os.environ.setdefault("PERPLEXITY_API_KEY", "dummy")

    async def dummy_openai_response(*args, **kwargs):
        return schemas.IndividualAIResponse(source="OpenAI", response="openai draft")

    async def dummy_claude_response(*args, **kwargs):
        return schemas.IndividualAIResponse(source="Claude", response="claude review")

    async def dummy_cohere_response(*args, **kwargs):
        return schemas.IndividualAIResponse(source="Cohere", response="cohere improve")

    async def dummy_gemini_response(*args, **kwargs):
        return schemas.IndividualAIResponse(source="Gemini", response="gemini answer")

    async def dummy_perplexity_response(*args, **kwargs):
        year = datetime.utcnow().year
        return schemas.IndividualAIResponse(source="Perplexity", response=f"info {year}")

    monkeypatch.setattr(main, "get_openai_response", dummy_openai_response)
    monkeypatch.setattr(main, "get_claude_response", dummy_claude_response)
    monkeypatch.setattr(main, "get_cohere_response", dummy_cohere_response)
    monkeypatch.setattr(main, "get_gemini_response", dummy_gemini_response)
    monkeypatch.setattr(main, "get_perplexity_response", dummy_perplexity_response)

    req = Request({"type": "http"})
    shell = schemas.CollaborativeResponseV2(prompt="q")

    res = await main.run_balance_mode_flow(
        original_prompt="最新のAIニュースは？",
        response_shell=shell,
        chat_history_for_ai=[],
        initial_user_prompt_for_session="AIについて",
        request=req,
        user_memories=None,
    )

    current_year = str(datetime.utcnow().year)
    assert current_year in res.step4_comprehensive_answer_perplexity.response
