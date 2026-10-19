"""検索したチャンクを根拠に、Gemini で回答を生成するモジュール。

ポイント:
  - プロンプトで「与えた資料だけを根拠に答える」「無ければ『わからない』と言う」と
    指示することで、もっともらしい嘘（ハルシネーション）を減らします。
  - API キーが無い場合は LLM を使わず、検索結果をそのまま見せる
    「抽出モード」で動きます（費用ゼロでデモできるようにするため）。
"""

from __future__ import annotations

from src.config import GEMINI_MODELS
from src.retriever import RetrievedChunk

SYSTEM_PROMPT = """あなたは社内ドキュメントに関する質問に答えるアシスタントです。
以下のルールを必ず守ってください。
1. 回答は【参考資料】に書かれている内容だけを根拠にしてください。あなた自身の知識で補ってはいけません。
2. 【参考資料】に答えが書かれていない場合は、推測せずに「わからない」とだけ答えてください。
3. 回答の文末には、根拠にした資料の番号を [1] [2] のように付けてください。
4. 日本語で、簡潔かつ丁寧に答えてください。"""


def format_citation(item: RetrievedChunk) -> str:
    """出典を「ファイル名 p.ページ」の形式で表す。"""
    return f"{item.chunk.source} p.{item.chunk.page}"


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    """検索結果に番号を振って並べ、質問と合わせたプロンプトを作る。"""
    context_text = "\n\n".join(
        f"[{i}] （出典: {format_citation(item)}）\n{item.chunk.text}" for i, item in enumerate(contexts, start=1)
    )
    return f"【参考資料】\n{context_text}\n\n【質問】\n{question}"


def generate_answer(
    question: str,
    contexts: list[RetrievedChunk],
    api_key: str,
    model: str = GEMINI_MODELS[0],
) -> str:
    """Gemini に問い合わせて、資料に基づいた回答を生成する。"""
    if not contexts:
        return "わからない（関連する資料が見つかりませんでした）"

    # google-genai は公式の新しい SDK（旧 google-generativeai の後継）
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(question, contexts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,  # 低めにして、資料に忠実で安定した回答にする
        ),
    )
    return (response.text or "").strip() or "わからない（回答を生成できませんでした）"


def extractive_answer(contexts: list[RetrievedChunk], max_items: int = 2) -> str:
    """API キーが無いときの代替回答。関連度の高いチャンク上位 max_items 件をそのまま示す。

    残りのチャンクは画面の「検索されたチャンク」欄で確認できます。
    """
    if not contexts:
        return "関連する箇所が見つかりませんでした。"
    lines = ["**最も関連性の高い箇所（抽出モード）**", ""]
    for i, item in enumerate(contexts[:max_items], start=1):
        lines.append(f"**[{i}] {format_citation(item)}**（類似度 {item.score:.3f}）")
        quoted = item.chunk.text.replace("\n", "\n> ")  # 複数行でも引用表示にする
        lines.append(f"> {quoted}")
        lines.append("")
    return "\n".join(lines)
