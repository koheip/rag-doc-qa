"""retriever（検索）のテスト。

本物の埋め込みモデルは読み込みに時間がかかるため、文字の出現回数から
ベクトルを作る簡単な偽物（FakeEmbedder）を使って高速にテストします。
"""

import pytest

from src.chunker import Chunk
from src.retriever import DenseRetriever, create_retriever
from tests.fakes import FakeEmbedder


@pytest.fixture
def retriever():
    chunks = [
        Chunk("有給休暇は3営業日前までに申請します。", "rules.pdf", 2, 0),
        Chunk("給与は毎月25日に振り込まれます。", "rules.pdf", 3, 1),
        Chunk("健康診断は毎年10月に実施します。", "faq.pdf", 1, 2),
    ]
    r = DenseRetriever(FakeEmbedder())
    r.index(chunks)
    return r


def test_index_counts_chunks(retriever):
    assert retriever.num_chunks == 3


def test_retrieve_returns_most_relevant_first(retriever):
    results = retriever.retrieve("有給休暇の申請", top_k=3)
    assert results[0].chunk.source == "rules.pdf"
    assert results[0].chunk.page == 2


def test_scores_are_sorted_and_in_range(retriever):
    results = retriever.retrieve("給与の振込日", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)  # コサイン類似度の範囲


def test_top_k_is_capped_by_number_of_chunks(retriever):
    assert len(retriever.retrieve("健康診断", top_k=10)) == 3
    assert len(retriever.retrieve("健康診断", top_k=1)) == 1


def test_retrieve_before_index_raises():
    with pytest.raises(RuntimeError):
        DenseRetriever(FakeEmbedder()).retrieve("質問")


def test_index_empty_raises():
    with pytest.raises(ValueError):
        DenseRetriever(FakeEmbedder()).index([])


def test_create_retriever_dense():
    assert isinstance(create_retriever("dense", FakeEmbedder()), DenseRetriever)


def test_create_retriever_unknown_strategy():
    with pytest.raises(ValueError):
        create_retriever("unknown", FakeEmbedder())
