"""Gemini を使った「検索 → 回答生成」までを画面なしで確認するスクリプト。

環境変数 GEMINI_API_KEY（または .env）が必要です。結果は docs/demo_output_gemini.txt に保存します。
※ API キーの値は画面にもファイルにも出力しません。

使い方:
    python scripts/demo_gemini.py                 # デフォルトモデル
    python scripts/demo_gemini.py gemini-3.8-flash  # モデルを指定
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from scripts.demo_pipeline import QUESTIONS  # noqa: E402
from src.chunker import chunk_pages  # noqa: E402
from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, GEMINI_MODELS  # noqa: E402
from src.embedder import E5Embedder  # noqa: E402
from src.llm import format_citation, generate_answer  # noqa: E402
from src.llm_client import GeminiClient  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.retriever import create_retriever  # noqa: E402

OUTPUT = ROOT / "docs" / "demo_output_gemini.txt"


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY が設定されていません")
    model = sys.argv[1] if len(sys.argv) > 1 else GEMINI_MODELS[0]

    pages = load_pdf(ROOT / "sample_docs" / "sample_company_rules.pdf")
    retriever = create_retriever("dense", E5Embedder())
    retriever.index(chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP))

    lines = [f"モデル: {model} / top-k: {DEFAULT_TOP_K} / チャンク: {DEFAULT_CHUNK_SIZE}文字（重なり{DEFAULT_CHUNK_OVERLAP}）", ""]
    for q in QUESTIONS:
        contexts = retriever.retrieve(q, top_k=DEFAULT_TOP_K)
        start = time.time()
        answer = generate_answer(q, contexts, GeminiClient(api_key, model)).text
        lines += [
            "=" * 70,
            f"Q: {q}",
            f"A: {answer}",
            "検索されたチャンク: " + " / ".join(f"[{i}] {format_citation(c)} ({c.score:.3f})" for i, c in enumerate(contexts, 1)),
            f"（生成 {time.time() - start:.1f} 秒）",
        ]
    text = "\n".join(lines).replace(api_key, "***")  # 念のためキーが混ざらないようにする
    OUTPUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
