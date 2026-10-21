"""LLM-as-a-judge（LLM による採点）のモジュール。

回答の正しさは「完全一致」では測れない（同じ内容でも言い回しが変わる）ため、
別の LLM に「模範解答と比べて正しいか」を判定させます。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm_client import LLMClient, LLMResponse

JUDGE_SYSTEM_PROMPT = """あなたは社内規則Q&Aシステムの回答を採点する、厳格で公平な採点者です。
【質問】【模範解答】【評価対象の回答】を読み、評価対象の回答が正しいかを判定してください。

採点基準:
- score = 1: 模範解答の要点（数値・期限・条件・可否など、質問に直接答える部分）を正しく含み、模範解答と矛盾する誤った情報を含まない。
  - 言い回しの違いは問わない。模範解答にない補足情報があっても、誤りでなければ減点しない。
  - 模範解答の補足的な条件（例外規定など）が抜けていても、質問への直接の答えが正しければ 1 とする。
- score = 0: 要点が欠けている、数値や条件が誤っている、誤った情報を含む、または「わからない」と答えている。

出力は次の JSON のみ（前後に文章を付けない）:
{"score": 0 または 1, "rationale": "判定理由を日本語で1〜2文"}"""


@dataclass
class JudgeResult:
    score: int  # 0 または 1
    rationale: str
    response: LLMResponse


def build_judge_prompt(question: str, reference: str, answer: str) -> str:
    return f"【質問】\n{question}\n\n【模範解答】\n{reference}\n\n【評価対象の回答】\n{answer}"


def parse_judge_output(text: str) -> tuple[int, str]:
    """採点 LLM の出力（JSON）から score と理由を取り出す。壊れた出力でもできるだけ読む。"""
    try:
        data = json.loads(text)
        return (1 if int(data["score"]) >= 1 else 0), str(data.get("rationale", ""))
    except (ValueError, KeyError, TypeError):
        match = re.search(r'"?score"?\s*[:：]\s*([01])', text)
        if match:
            return int(match.group(1)), text.strip()
        raise ValueError(f"採点結果を読み取れませんでした: {text[:100]}") from None


def judge_answer(question: str, reference: str, answer: str, judge: LLMClient) -> JudgeResult:
    """回答を採点する（temperature=0 で、なるべく毎回同じ判定になるようにする）。"""
    response = judge.generate(
        build_judge_prompt(question, reference, answer),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        json_mode=True,
    )
    score, rationale = parse_judge_output(response.text)
    return JudgeResult(score=score, rationale=rationale, response=response)
