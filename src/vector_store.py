"""ベクトルを保存し、似ているベクトルを高速に探す「ベクトルストア」モジュール。

FAISS（Meta 製の近傍探索ライブラリ）の IndexFlatIP を使います。
  - Flat: 全件を総当たりで比較する（近似なし＝結果が正確）
  - IP  : 内積（Inner Product）で比較する
ベクトルを長さ1に正規化しておけば、内積はコサイン類似度と同じになります。
"""

from __future__ import annotations

import faiss
import numpy as np

from src.chunker import Chunk


class FaissVectorStore:
    """チャンクとそのベクトルをセットで管理するクラス。"""

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []  # index の i 番目のベクトル ⇔ chunks[i]

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """ベクトルとチャンクを追加する（順番を対応させて保存）。"""
        if len(embeddings) != len(chunks):
            raise ValueError("ベクトルとチャンクの件数が一致しません")
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)  # 念のため正規化（コサイン類似度にするため）
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """質問ベクトルに近い順に (チャンク, 類似度) を top_k 件返す。"""
        if len(self) == 0:
            return []
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(top_k, len(self))
        scores, indices = self.index.search(query, k)
        return [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
