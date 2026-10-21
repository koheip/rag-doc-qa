"""llm モジュールのうち、API を呼ばない部分（プロンプト作成・抽出モード）のテスト。"""

from src.chunker import Chunk
from src.llm import build_prompt, extractive_answer
from src.retriever import RetrievedChunk


def _ctx():
    return [RetrievedChunk(Chunk("給与は25日払いです。", "rules.pdf", 3, 0), 0.9)]


def test_build_prompt_contains_context_and_citation():
    prompt = build_prompt("給与日は？", _ctx())
    assert "給与は25日払いです。" in prompt
    assert "rules.pdf p.3" in prompt
    assert "給与日は？" in prompt


def test_extractive_answer_shows_citation():
    assert "rules.pdf p.3" in extractive_answer(_ctx())
    assert "見つかりませんでした" in extractive_answer([])


def test_cited_indices_extracts_valid_numbers():
    from src.llm import cited_indices

    assert cited_indices("25日です [2]。また [1][2] も参照。", num_contexts=4) == [1, 2]
    assert cited_indices("範囲外 [9] と [0]", num_contexts=4) == []
    assert cited_indices("わからない", num_contexts=4) == []


def test_is_unknown_ignores_citations():
    from src.llm import UNKNOWN_ANSWER, cited_indices, is_unknown

    assert is_unknown("わからない [1][2][3][4]。")
    assert is_unknown("わからないです。")
    assert not is_unknown("給料日は25日です [2]。")
    assert not is_unknown("わからない点は人事部に確認してください [1]。")
    assert cited_indices(UNKNOWN_ANSWER, num_contexts=4) == []


def test_generate_answer_uses_llm_client_and_keeps_usage():
    from src.llm import generate_answer
    from tests.fakes import FakeLLM

    llm = FakeLLM("給与は25日払いです [1]。")
    response = generate_answer("給与日は？", _ctx(), llm)
    assert response.text == "給与は25日払いです [1]。"
    assert (response.input_tokens, response.output_tokens) == (100, 10)
    assert "rules.pdf p.3" in llm.prompts[0]  # 出典付きの資料がプロンプトに入っている


def test_generate_answer_normalizes_unknown():
    from src.llm import UNKNOWN_ANSWER, generate_answer
    from tests.fakes import FakeLLM

    assert generate_answer("食堂は？", _ctx(), FakeLLM("わからない [1][2]")).text == UNKNOWN_ANSWER
    assert generate_answer("食堂は？", [], FakeLLM("呼ばれない")).text == UNKNOWN_ANSWER


def test_suggested_retry_delay_is_parsed_from_error_details():
    from types import SimpleNamespace

    from src.llm_client import _suggested_retry_delay_s

    error = SimpleNamespace(details={"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "84826s"}]}})
    assert _suggested_retry_delay_s(error) == 84826.0
    assert _suggested_retry_delay_s(SimpleNamespace(details=None)) is None
