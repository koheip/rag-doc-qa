"""chunker（チャンク分割）のテスト。"""

import pytest

from src.chunker import chunk_pages, split_sentences, split_text
from src.loader import Page, clean_text


def test_split_sentences_japanese():
    text = "今日は晴れです。明日は雨でしょうか？傘を持っていきます！"
    assert split_sentences(text) == ["今日は晴れです。", "明日は雨でしょうか？", "傘を持っていきます！"]


def test_short_text_is_single_chunk():
    assert split_text("短い文章です。", chunk_size=100, overlap=10) == ["短い文章です。"]


def test_chunks_respect_size_limit():
    text = "これはテスト用の文です。" * 50  # 1文12文字 × 50
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_cover_all_sentences():
    sentences = [f"文{i:02d}の内容です。" for i in range(30)]
    chunks = split_text("".join(sentences), chunk_size=60, overlap=0)
    joined = "".join(chunks)
    for s in sentences:
        assert s in joined  # どの文も取りこぼしていない


def test_overlap_carries_previous_sentence():
    sentences = [f"文{i:02d}の内容です。" for i in range(10)]  # 1文9文字
    chunks = split_text("".join(sentences), chunk_size=30, overlap=10)
    # 前のチャンクの最後の文が、次のチャンクの先頭に含まれている
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev)[-1]
        assert nxt.startswith(last_sentence)


def test_long_sentence_is_force_split():
    long_sentence = "あ" * 250  # 句点の無い長文
    chunks = split_text(long_sentence, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 3


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunk_pages_keeps_source_and_page():
    pages = [Page("a.pdf", 1, "一ページ目です。"), Page("a.pdf", 2, "二ページ目です。")]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert [(c.source, c.page) for c in chunks] == [("a.pdf", 1), ("a.pdf", 2)]
    assert [c.chunk_id for c in chunks] == [0, 1]


def test_clean_text_joins_broken_japanese_lines():
    raw = "有給休暇は3営業日前までに申\n請すること。\n次の文です。"
    assert clean_text(raw) == "有給休暇は3営業日前までに申請すること。\n次の文です。"


def test_clean_text_removes_page_number_footer():
    assert clean_text("本文です。\n- 2 / 6 -") == "本文です。"
