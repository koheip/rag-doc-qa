"""評価指標（eval/metrics.py）と採点結果の読み取り（eval/judge.py）のテスト。"""

import math

import pytest

from eval.judge import parse_judge_output
from eval.metrics import first_relevant_rank, hit_at_k, percentile, recall_at_k, reciprocal_rank
from eval.pricing import estimate_cost_usd
from src.chunker import Chunk

RETRIEVED = [
    Chunk("給与は翌月25日に支払う。", "a.pdf", 3, 0),
    Chunk("有給休暇は3営業日前まで。", "a.pdf", 2, 1),
    Chunk("結婚時には5万円を支給。", "a.pdf", 7, 2),
]


def test_rank_hit_and_mrr():
    evidence = [{"page": 2, "keyword": "3営業日前"}]
    rank = first_relevant_rank(RETRIEVED, evidence)
    assert rank == 2
    assert hit_at_k(rank, 1) == 0.0
    assert hit_at_k(rank, 3) == 1.0
    assert reciprocal_rank(rank) == 0.5


def test_evidence_must_match_page_and_keyword():
    # キーワードが合っていてもページが違えば関連チャンクとはみなさない
    assert first_relevant_rank(RETRIEVED, [{"page": 9, "keyword": "3営業日前"}]) is None
    assert reciprocal_rank(None) == 0.0
    assert hit_at_k(None, 3) == 0.0


def test_keyword_match_ignores_whitespace():
    assert first_relevant_rank(RETRIEVED, [{"page": 3, "keyword": "翌月 25日"}]) == 1


def test_recall_with_multiple_evidence():
    evidence = [{"page": 7, "keyword": "5万円"}, {"page": 2, "keyword": "結婚するときは5日"}]
    assert recall_at_k(RETRIEVED, evidence, k=3) == 0.5
    assert recall_at_k(RETRIEVED, evidence, k=1) == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1.0, 2.0], 95) == pytest.approx(1.95)
    assert math.isnan(percentile([], 50))


def test_cost_estimate():
    # 入力 100万 + 出力 100万トークン = $0.30 + $2.50
    assert estimate_cost_usd("gemini-3.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(2.80)
    assert estimate_cost_usd("unknown-model", 10, 10) is None


def test_parse_judge_output():
    assert parse_judge_output('{"score": 1, "rationale": "正しい"}') == (1, "正しい")
    assert parse_judge_output('{"score": 0, "rationale": "数値が違う"}')[0] == 0
    assert parse_judge_output('結果: "score": 1')[0] == 1  # JSON が壊れていても読む
    with pytest.raises(ValueError):
        parse_judge_output("採点できません")
