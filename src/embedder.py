"""テキストをベクトル（数値の並び）に変換する「埋め込み（Embedding）」モジュール。

意味が近い文章ほど、ベクトル同士のコサイン類似度が高くなります。
これを使って「質問に意味的に近いチャンク」を探します。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.config import EMBEDDING_MODEL_NAME


class Embedder(Protocol):
    """埋め込みモデルが満たすべきインターフェース。

    この形さえ守れば中身は何でもよいので、テストでは軽量な偽物（Fake）に差し替えられます。
    """

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """文書（チャンク）のリストを (件数, 次元数) の行列に変換する。"""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """質問文を (次元数,) のベクトルに変換する。"""
        ...


class E5Embedder:
    """sentence-transformers で multilingual-e5 を使う埋め込みクラス。

    E5 系モデルは学習時の約束として、
      - 検索される側（文書）には "passage: "
      - 検索する側（質問）には "query: "
    を先頭に付ける必要があります。付けないと精度が落ちます。
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        # 重いライブラリなので、実際に使うときだけ import する（テストを速くするため）
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        inputs = [f"passage: {t}" for t in texts]
        # normalize_embeddings=True で長さ1に正規化 → 内積 = コサイン類似度 になる
        vectors = self.model.encode(inputs, normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([f"query: {text}"], normalize_embeddings=True)
        return np.asarray(vector[0], dtype="float32")
