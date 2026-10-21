"""モデルごとの料金表（費用の見積もりに使う）。

出典: Gemini Developer API pricing — https://ai.google.dev/gemini-api/docs/pricing
確認日: 2026-10-21（ページの最終更新 2026-10-19 UTC）
※ Standard（通常）プランの Paid Tier の料金。料金は変わることがあるので、使う前に出典を確認してください。
※ 無料枠（Free Tier）では、どちらのモデルも入力・出力とも無料（$0）。
"""

from __future__ import annotations

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_CHECKED_DATE = "2026-10-21"

# USD / 100万トークン。出力料金には思考（thinking）トークンも含まれる。
PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    # 3.6 Flash / 3.8 Flash は 2026-12-31 までの料金（2027-01-01 から入力 $1.50 / 出力 $7.50 に値上げ予定）
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    # 3.8 Flash は 2026-12-31 までの料金（2027-01-01 から入力 $1.50 / 出力 $7.50 に値上げ予定）
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}

# 無料枠で利用できるモデル（無料枠の費用は $0 として表示する）
# ※ 無料枠には1日あたりのリクエスト数上限がある（例: 2026-10-21 時点で gemini-3.6-flash は 20回/日 だった）
FREE_TIER_MODELS = {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """有料枠（Paid Tier）で同じ処理をした場合の費用（USD）。料金表に無いモデルは None。"""
    price = PRICING_USD_PER_1M_TOKENS.get(model)
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
