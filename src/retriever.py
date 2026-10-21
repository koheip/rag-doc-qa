"""質問に関連するチャンクを探す「検索器（Retriever）」のモジュール。

検索のやり方（戦略）を差し替えられるように、次の形にしています。
  - Retriever     : 検索器が満たすべきインターフェース（Protocol）
  - DenseRetriever: 埋め込みベクトルで意味の近さを測る「密ベクトル検索」（現在の標準）
  - create_retriever(): 戦略名（"dense" / "hybrid"）から検索器を作る工場関数

アプリや評価スクリプトは create_retriever() だけを使うので、
新しい検索方法（例: BM25 とのハイブリッド検索）を追加しても呼び出し側は変わりません。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.chunker import Chunk
from src.embedder import Embedder
from src.vector_store import FaissVectorStore

# 選べる検索戦略。"hybrid" は src/bm25.py の課題を実装すると使えるようになります。
RETRIEVAL_STRATEGIES = ("dense", "hybrid")


@dataclass
class RetrievedChunk:
    """検索結果1件分（チャンク＋スコア）。"""

    chunk: Chunk
    score: float  # dense ならコサイン類似度（1に近いほど似ている）。戦略によって尺度は異なる


class Retriever(Protocol):
    """検索器のインターフェース。この3つを持っていれば、どんな検索方法でも差し替え可能。"""

    @property
    def num_chunks(self) -> int: ...

    def index(self, chunks: list[Chunk]) -> None:
        """チャンクを検索できる状態に登録する。"""
        ...

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """質問に関連するチャンクを、関連度の高い順に top_k 件返す。"""
        ...


class DenseRetriever:
    """埋め込みモデル＋ベクトルストアによる「密ベクトル検索」。

    意味が近ければ言い回しが違っても見つけられるのが長所。
    逆に、条文番号や固有名詞など「文字が一致すること」が大事な検索は苦手なことがある。
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.store: FaissVectorStore | None = None

    @property
    def num_chunks(self) -> int:
        return len(self.store) if self.store else 0

    def index(self, chunks: list[Chunk]) -> None:
        """チャンクをベクトル化してベクトルストアに登録する（既存の内容は作り直す）。"""
        if not chunks:
            raise ValueError("インデックスするチャンクがありません")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store = FaissVectorStore(dim=embeddings.shape[1])
        self.store.add(embeddings, chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """質問に意味が近いチャンクを、類似度の高い順に top_k 件返す。"""
        if self.store is None:
            raise RuntimeError("先に index() でドキュメントを登録してください")
        query_vector = self.embedder.embed_query(query)
        return [RetrievedChunk(chunk=c, score=s) for c, s in self.store.search(query_vector, top_k)]


def create_retriever(strategy: str, embedder: Embedder) -> Retriever:
    """戦略名から検索器を作る。

    Args:
        strategy: "dense"（密ベクトル検索）または "hybrid"（BM25 ＋ 密ベクトル）
        embedder: 埋め込みモデル
    """
    if strategy == "dense":
        return DenseRetriever(embedder)
    if strategy == "hybrid":
        # 循環 import を避けるため、使うときだけ読み込む
        from src.bm25 import HybridRetriever

        return HybridRetriever(DenseRetriever(embedder))
    raise ValueError(f"未対応の検索戦略です: {strategy}（選択肢: {', '.join(RETRIEVAL_STRATEGIES)}）")
