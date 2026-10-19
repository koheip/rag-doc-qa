"""埋め込みモデルとベクトルストアを組み合わせて「検索器」を作るモジュール。

アプリ側はこのクラスだけを触ればよいようにして、内部の部品
（どの埋め込みモデルか、どのベクトルストアか）を意識しなくて済むようにしています。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunker import Chunk
from src.embedder import Embedder
from src.vector_store import FaissVectorStore


@dataclass
class RetrievedChunk:
    """検索結果1件分（チャンク＋類似度スコア）。"""

    chunk: Chunk
    score: float  # コサイン類似度（1に近いほど似ている）


class Retriever:
    """チャンクをインデックス化し、質問に関連するチャンクを検索する。"""

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
