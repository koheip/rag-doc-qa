"""評価指標を計算する関数（API やモデルに依存しない純粋な関数なので、単体テストしやすい）。

用語:
  - evidence（根拠）: 正解が書かれている場所。「ページ番号＋その箇所に含まれるキーワード」で表す
  - 関連チャンク    : evidence のどれかを含むチャンク
  - rank（順位）    : 検索結果の何番目か（1始まり）
"""

from __future__ import annotations

import math
import re

from src.chunker import Chunk

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """比較のため空白・改行を取り除く（PDF 由来の改行位置の違いを無視する）。"""
    return _SPACES.sub("", text)


def matches_evidence(chunk: Chunk, evidence: dict) -> bool:
    """チャンクが evidence（{"page": int, "keyword": str}）を含むかどうか。"""
    return chunk.page == evidence["page"] and normalize(evidence["keyword"]) in normalize(chunk.text)


def first_relevant_rank(retrieved: list[Chunk], evidences: list[dict]) -> int | None:
    """最初に関連チャンクが現れた順位（1始まり）。見つからなければ None。"""
    for rank, chunk in enumerate(retrieved, start=1):
        if any(matches_evidence(chunk, ev) for ev in evidences):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    """上位 k 件の中に関連チャンクが1つでもあれば 1、なければ 0。"""
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(retrieved: list[Chunk], evidences: list[dict], k: int) -> float:
    """必要な根拠（evidence）のうち、上位 k 件でカバーできた割合（0〜1）。

    根拠が複数ある質問（例: 「休暇日数」と「お祝い金」の両方が必要）で、片方しか取れていない状態を測れる。
    """
    if not evidences:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for ev in evidences if any(matches_evidence(c, ev) for c in top))
    return found / len(evidences)


def reciprocal_rank(rank: int | None) -> float:
    """MRR の1問分。1位なら 1.0、2位なら 0.5、見つからなければ 0。"""
    return 1.0 / rank if rank else 0.0


def percentile(values: list[float], p: float) -> float:
    """p パーセンタイル（線形補間）。p50 = 中央値、p95 = 遅い方から5%の境目。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(pos), math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
