"""【課題用】src/bm25.py（BM25 ハイブリッド検索）のテスト。

実装前はすべて失敗します。通常の `pytest` では実行されず、次のコマンドで実行します:
    python -m pytest -m exercise -v
"""

import pytest

from src.chunker import Chunk
from src.retriever import DenseRetriever, create_retriever
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.exercise


# ---------------------------------------------------------------- tokenize
def test_tokenize_character_bigrams():
    from src.bm25 import tokenize

    assert tokenize("給与の支払い") == ["給与", "与の", "の支", "支払", "払い"]


def test_tokenize_removes_spaces_and_punctuation_and_lowercases():
    from src.bm25 import tokenize

    assert tokenize("PC、壊れた。") == ["pc", "c壊", "壊れ", "れた"]
    assert tokenize("（給与） ") == ["給与"]


def test_tokenize_short_and_empty():
    from src.bm25 import tokenize

    assert tokenize("A") == ["a"]
    assert tokenize("。") == []
    assert tokenize("給与支払", n=3) == ["給与支", "与支払"]


# ---------------------------------------------------------------- BM25
DOCS = [
    "給与は毎月25日に支払う。",
    "有給休暇は3営業日前までに申請する。",
    "通勤手当の上限は月額3万円とする。",
    "内線1234に連絡してください。",
]


def test_bm25_scores_shape_and_best_match():
    from src.bm25 import BM25Index

    bm25 = BM25Index()
    bm25.fit(DOCS)
    scores = bm25.scores("内線1234")
    assert len(scores) == len(DOCS)
    assert max(range(len(DOCS)), key=lambda i: scores[i]) == 3


def test_bm25_no_overlap_gives_zero():
    from src.bm25 import BM25Index

    bm25 = BM25Index()
    bm25.fit(DOCS)
    assert bm25.scores("ドレスコード") == [0.0] * len(DOCS)


def test_bm25_rare_token_weighs_more_than_common_token():
    from src.bm25 import BM25Index

    # 「規則」は全文書にあるが、「賞与」は1文書だけ → 賞与を含む文書が最上位になるべき（IDF の効果）
    docs = ["規則の規則の規則。", "規則。賞与は年2回。", "規則。休暇。"]
    bm25 = BM25Index()
    bm25.fit(docs)
    scores = bm25.scores("規則と賞与")
    assert scores[1] == max(scores)


def test_bm25_longer_document_is_penalized():
    from src.bm25 import BM25Index

    # 同じ回数「賞与」を含むなら、短い文書の方がスコアが高い（文書長の正規化 b の効果）
    docs = ["賞与。", "賞与。" + "その他の長い説明文。" * 10, "休暇。"]
    bm25 = BM25Index()
    bm25.fit(docs)
    scores = bm25.scores("賞与")
    assert scores[0] > scores[1] > 0


# ---------------------------------------------------------------- RRF
def test_rrf_combines_rankings():
    from src.bm25 import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[2, 0, 1], [0, 2, 3]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids) == {0, 1, 2, 3}
    assert ids[:2] in ([0, 2], [2, 0])  # 両方で上位の 0 と 2 が上に来る
    assert dict(fused)[0] == pytest.approx(1 / 62 + 1 / 61)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------- HybridRetriever
CHUNKS = [
    Chunk("第9条 給与は毎月25日に支払う。", "rules.pdf", 3, 0),
    Chunk("第6条 有給休暇は3営業日前までに申請する。", "rules.pdf", 2, 1),
    Chunk("Q. パソコンが故障したら？ A. 内線1234に連絡。", "faq.pdf", 1, 2),
    Chunk("第21条 退職届は1か月前までに提出する。", "rules.pdf", 5, 3),
]


def test_hybrid_retriever_via_factory():
    retriever = create_retriever("hybrid", FakeEmbedder())
    retriever.index(CHUNKS)
    assert retriever.num_chunks == 4
    results = retriever.retrieve("内線1234", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == 2
    assert results[0].score >= results[1].score


def test_hybrid_top_k_and_errors():
    from src.bm25 import HybridRetriever

    retriever = HybridRetriever(DenseRetriever(FakeEmbedder()))
    with pytest.raises(RuntimeError):
        retriever.retrieve("給与")
    with pytest.raises(ValueError):
        retriever.index([])
    retriever.index(CHUNKS)
    assert len(retriever.retrieve("給与", top_k=10)) == 4
