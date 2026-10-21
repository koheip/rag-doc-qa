"""RAG パイプラインを画面なしで一通り動かす確認用スクリプト（API キー不要）。

サンプル PDF を「読み込み → チャンク分割 → ベクトル化 → 検索」し、
いくつかの質問に対して取り出されたチャンクを表示します。

使い方:
    python scripts/demo_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_pages  # noqa: E402
from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE  # noqa: E402
from src.embedder import E5Embedder  # noqa: E402
from src.llm import extractive_answer, format_citation  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.retriever import create_retriever  # noqa: E402

QUESTIONS = [
    "有給休暇は何日前までに申請する必要がありますか？",
    "給料日はいつですか？",
    "パソコンが壊れたらどこに連絡すればいい？",
    "リモートワークは週に何日までできますか？",
    "退職するときはいつまでに伝えればいい？",
    "生成AIを仕事で使ってもいいですか？",
    "社員食堂のメニューを教えて",  # 資料に無い質問（類似度が低くなるはず）
]


def main() -> None:
    pdf_path = ROOT / "sample_docs" / "sample_company_rules.pdf"
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    print(f"ページ数: {len(pages)} / チャンク数: {len(chunks)}")

    start = time.time()
    retriever = create_retriever("dense", E5Embedder())
    retriever.index(chunks)
    print(f"インデックス作成: {time.time() - start:.1f} 秒\n")

    for q in QUESTIONS:
        results = retriever.retrieve(q, top_k=3)
        print("=" * 70)
        print(f"Q: {q}")
        for i, r in enumerate(results, start=1):
            preview = r.chunk.text.replace("\n", " ")[:80]
            print(f"  [{i}] {format_citation(r)}  score={r.score:.3f}  {preview}...")
    print("=" * 70)
    print("\n--- 抽出モードの回答例（1問目）---")
    print(extractive_answer(retriever.retrieve(QUESTIONS[0], top_k=3)))


if __name__ == "__main__":
    main()
