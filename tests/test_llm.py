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
