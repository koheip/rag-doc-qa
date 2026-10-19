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
