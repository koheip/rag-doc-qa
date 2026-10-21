"""RAG の評価スクリプト。検索精度・回答品質・費用・応答時間を測り、Markdown と JSON に保存します。

使い方（プロジェクト直下で実行）:
    # 検索精度だけ（API キー不要・無料）
    python eval/run_eval.py --skip-generation

    # 検索＋回答生成＋LLM による採点（GEMINI_API_KEY が必要）
    python eval/run_eval.py

    # モデル比較・チャンクサイズ比較・検索方式の比較
    python eval/run_eval.py --models gemini-3.5-flash-lite gemini-3.8-flash
    python eval/run_eval.py --chunk-sizes 150 200 300 --skip-generation
    python eval/run_eval.py --retrievers dense hybrid --skip-generation   # hybrid は課題の実装後

検索精度は「--retrievers × --chunk-sizes」のすべての組み合わせで測ります。
回答生成は API 呼び出し回数を抑えるため、最初の組み合わせ（先頭の retriever・chunk size）だけで、
--models に指定した各モデルについて実行します。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from eval.judge import judge_answer  # noqa: E402
from eval.metrics import (  # noqa: E402
    first_relevant_rank,
    hit_at_k,
    matches_evidence,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from eval.pricing import (  # noqa: E402
    FREE_TIER_MODELS,
    PRICING_CHECKED_DATE,
    PRICING_SOURCE_URL,
    estimate_cost_usd,
)
from src.chunker import Chunk, chunk_pages  # noqa: E402
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, GEMINI_MODELS  # noqa: E402
from src.embedder import E5Embedder  # noqa: E402
from src.llm import generate_answer, is_unknown  # noqa: E402
from src.llm_client import GeminiClient  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.retriever import Retriever, create_retriever  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEFAULT_PDF = ROOT / "sample_docs" / "sample_company_rules.pdf"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "results"
RANK_DEPTH = 10  # MRR などのために上位何件まで検索するか


# ---------------------------------------------------------------------------
# データの読み込みと検証
# ---------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_dataset(questions: list[dict], chunks: list[Chunk]) -> list[str]:
    """各 evidence が、どこかのチャンクに実在するかを確認する（データセットの書き間違い検出）。"""
    problems = []
    for q in questions:
        for ev in q["evidence"]:
            if not any(matches_evidence(c, ev) for c in chunks):
                problems.append(f"{q['id']}: p.{ev['page']} の「{ev['keyword']}」がどのチャンクにも見つかりません")
    return problems


# ---------------------------------------------------------------------------
# 検索の評価
# ---------------------------------------------------------------------------
def evaluate_retrieval(retriever: Retriever, questions: list[dict], top_k: int) -> dict:
    """回答可能な質問について Hit@1 / Hit@3 / Recall@k / MRR を計算する。"""
    rows = []
    for q in questions:
        if not q["evidence"]:  # 回答不能な質問は検索の評価から除外
            continue
        retrieved = [r.chunk for r in retriever.retrieve(q["question"], top_k=RANK_DEPTH)]
        rank = first_relevant_rank(retrieved, q["evidence"])
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "rank": rank,
                "hit@1": hit_at_k(rank, 1),
                "hit@3": hit_at_k(rank, 3),
                f"recall@{top_k}": recall_at_k(retrieved, q["evidence"], top_k),
                "rr": reciprocal_rank(rank),
                "top_pages": [c.page for c in retrieved[:top_k]],
            }
        )
    summary = {
        "n": len(rows),
        "hit@1": mean([r["hit@1"] for r in rows]),
        "hit@3": mean([r["hit@3"] for r in rows]),
        f"recall@{top_k}": mean([r[f"recall@{top_k}"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "hit@3_by_type": {
            t: mean([r["hit@3"] for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})
        },
    }
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# 回答生成の評価
# ---------------------------------------------------------------------------
@dataclass
class GenerationRow:
    id: str
    type: str
    question: str
    answer: str
    answerable: bool
    correct: int | None  # 回答可能: 採点で正解なら1 / 回答不能: 正しく「わからない」と答えたら1 / 採点失敗は None
    refused: bool  # 「わからない」と答えたか
    rationale: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None
    judge_tokens: dict = field(default_factory=dict)


def evaluate_generation(
    retriever: Retriever,
    questions: list[dict],
    model: str,
    judge_model: str,
    api_key: str,
    top_k: int,
    sleep_s: float,
) -> dict:
    """各質問に回答を生成し、採点・トークン数・費用・応答時間を記録する。"""
    llm = GeminiClient(api_key, model, max_retries=4, initial_wait_s=10)
    judge = GeminiClient(api_key, judge_model, max_retries=4, initial_wait_s=10)
    rows: list[GenerationRow] = []

    for i, q in enumerate(questions, start=1):
        contexts = retriever.retrieve(q["question"], top_k=top_k)
        response = generate_answer(q["question"], contexts, llm)
        time.sleep(sleep_s)  # 無料枠のレート制限（1分あたりのリクエスト数）対策

        answerable = bool(q["evidence"])
        refused = is_unknown(response.text)
        row = GenerationRow(
            id=q["id"],
            type=q["type"],
            question=q["question"],
            answer=response.text,
            answerable=answerable,
            correct=0,
            refused=refused,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_s=response.latency_s,
            cost_usd=estimate_cost_usd(model, response.input_tokens, response.output_tokens),
        )
        if not answerable:
            row.correct = int(refused)
            row.rationale = "正しく回答を控えた" if refused else "資料に無いのに回答した（ハルシネーションの疑い）"
        elif refused:
            row.rationale = "資料に答えがあるのに「わからない」と答えた"
        else:
            try:
                result = judge_answer(q["question"], q["reference_answer"], response.text, judge)
                row.correct, row.rationale = result.score, result.rationale
                row.judge_tokens = {"input": result.response.input_tokens, "output": result.response.output_tokens}
            except Exception as e:  # 採点モデルが混雑していても評価全体は止めない（集計から除外）
                row.correct, row.rationale = None, f"採点に失敗（{type(e).__name__}）"
            time.sleep(sleep_s)
        rows.append(row)
        print(f"  [{model}] {i}/{len(questions)} {q['id']} correct={row.correct} ({row.latency_s:.1f}s)", flush=True)

    judged = [r for r in rows if r.correct is not None]  # 採点に失敗した行は正答率の計算から除く
    answerable_rows = [r for r in judged if r.answerable]
    unanswerable_rows = [r for r in judged if not r.answerable]
    costs = [r.cost_usd for r in rows if r.cost_usd is not None]
    latencies = [r.latency_s for r in rows if r.latency_s > 0]
    summary = {
        "model": model,
        "judge_model": judge_model,
        "n": len(rows),
        "accuracy_answerable": mean([r.correct for r in answerable_rows]),
        "false_refusal_rate": mean([float(r.refused) for r in rows if r.answerable]),
        "correct_refusal_rate": mean([r.correct for r in unanswerable_rows]),
        "overall_accuracy": mean([r.correct for r in judged]),
        "judge_failures": len(rows) - len(judged),
        "avg_input_tokens": mean([r.input_tokens for r in rows]),
        "avg_output_tokens": mean([r.output_tokens for r in rows]),
        "cost_per_question_usd_paid": mean(costs) if costs else None,
        "cost_per_1000_questions_usd_paid": mean(costs) * 1000 if costs else None,
        "free_tier": model in FREE_TIER_MODELS,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }
    return {"summary": summary, "rows": [asdict(r) for r in rows]}


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------
def pct(x: float) -> str:
    return "-" if x != x else f"{x * 100:.1f}%"  # x != x は NaN（該当する質問が無い）の判定


def build_markdown(meta: dict, retrieval: dict, generation: dict, top_k: int) -> str:
    lines = [
        f"# RAG 評価結果（{meta['timestamp']}）",
        "",
        f"- データセット: `{meta['questions_file']}`（{meta['n_questions']} 問：回答可能 {meta['n_answerable']} / 回答不能 {meta['n_unanswerable']}）",
        f"- 文書: `{meta['pdf']}`",
        f"- top-k: {top_k}（Recall@{top_k} と回答生成に使用）、MRR は上位 {RANK_DEPTH} 件で計算",
        f"- 埋め込みモデル: `{meta['embedding_model']}`",
        "",
        "## 検索精度（回答可能な質問のみ）",
        "",
        f"| 検索方式 | チャンク | チャンク数 | Hit@1 | Hit@3 | Recall@{top_k} | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, res in retrieval.items():
        s = res["summary"]
        lines.append(
            f"| {res['strategy']} | {res['chunk_size']} | {res['num_chunks']} | {pct(s['hit@1'])} | {pct(s['hit@3'])} "
            f"| {pct(s[f'recall@{top_k}'])} | {s['mrr']:.3f} |"
        )
    types = sorted({t for res in retrieval.values() for t in res["summary"]["hit@3_by_type"]})
    lines += ["", "### 質問タイプ別 Hit@3", "", "| 検索方式 | チャンク | " + " | ".join(types) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in types) + " |"]
    for res in retrieval.values():
        by_type = res["summary"]["hit@3_by_type"]
        lines.append(f"| {res['strategy']} | {res['chunk_size']} | " + " | ".join(pct(by_type.get(t, float('nan'))) for t in types) + " |")

    first = next(iter(retrieval.values()))
    misses = [r for r in first["rows"] if r["hit@3"] == 0]
    if misses:
        lines += ["", f"### 上位3件で根拠を取れなかった質問（{first['strategy']} / {first['chunk_size']}）", ""]
        for r in misses:
            lines.append(f"- {r['id']}（{r['type']}）{r['question']} → 正解の順位: {r['rank'] or f'{RANK_DEPTH}位以内になし'}")

    if generation:
        lines += [
            "",
            f"## 回答品質・費用・速度（検索: {meta['gen_strategy']} / チャンク {meta['gen_chunk_size']}）",
            "",
            "| モデル | 正答率（回答可能） | 誤って拒否 | 正しく拒否（回答不能） | 全体正答率 | 平均入力tok | 平均出力tok "
            "| 費用/1000問（無料枠） | 費用/1000問（有料枠換算） | 応答 p50 | 応答 p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for res in generation.values():
            s = res["summary"]
            paid = f"${s['cost_per_1000_questions_usd_paid']:.3f}" if s["cost_per_1000_questions_usd_paid"] is not None else "-"
            free = "$0" if s["free_tier"] else "-"
            lines.append(
                f"| {s['model']} | {pct(s['accuracy_answerable'])} | {pct(s['false_refusal_rate'])} "
                f"| {pct(s['correct_refusal_rate'])} | {pct(s['overall_accuracy'])} | {s['avg_input_tokens']:.0f} "
                f"| {s['avg_output_tokens']:.0f} | {free} | {paid} | {s['latency_p50_s']:.2f}s | {s['latency_p95_s']:.2f}s |"
            )
        lines += [
            "",
            f"- 採点（LLM-as-a-judge）: `{next(iter(generation.values()))['summary']['judge_model']}`、0/1 の2値。採点の費用は上表に含めていない",
            f"- 料金: {PRICING_SOURCE_URL}（{PRICING_CHECKED_DATE} 確認、Standard の Paid Tier）。無料枠では $0",
            "- 応答時間: 回答生成の API 呼び出し1回の時間（検索時間・リトライ待ちは含まない）",
        ]
        for model, error in meta.get("failed_models", {}).items():
            lines.append(f"- ⚠️ `{model}` は評価できませんでした（{error}）")
        for res in generation.values():
            if res["summary"]["judge_failures"]:
                lines.append(f"- ⚠️ `{res['summary']['model']}`: 採点に失敗した {res['summary']['judge_failures']} 問は正答率から除外")
        for res in generation.values():
            wrong = [r for r in res["rows"] if r["correct"] == 0]
            if wrong:
                lines += ["", f"### 不正解の詳細（{res['summary']['model']}）", ""]
                for r in wrong:
                    answer = r["answer"].replace("\n", " ")
                    lines.append(f"- **{r['id']}** {r['question']}\n  - 回答: {answer}\n  - 判定理由: {r['rationale']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG の評価（検索精度・回答品質・費用・速度）")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--retrievers", nargs="+", default=["dense"], help="検索方式（dense / hybrid）")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[DEFAULT_CHUNK_SIZE])
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="オーバーラップ = チャンクサイズ × この値")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--models", nargs="+", default=[GEMINI_MODELS[0]], help="回答生成に使うモデル")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite", help="採点に使うモデル（回答モデルとは別のモデルを推奨）")
    parser.add_argument("--skip-generation", action="store_true", help="検索精度だけ測る（API キー不要）")
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 問だけ使う（動作確認用）")
    parser.add_argument("--sleep", type=float, default=4.0, help="API 呼び出しの間隔（秒）。無料枠のレート制限対策")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    questions = load_questions(args.questions)[: args.limit]
    pages = load_pdf(args.pdf)
    embedder = E5Embedder()

    # 1. 検索の評価（retriever × chunk size の全組み合わせ）
    retrieval: dict[str, dict] = {}
    retrievers: dict[str, Retriever] = {}
    for chunk_size in args.chunk_sizes:
        chunks = chunk_pages(pages, chunk_size, int(chunk_size * args.overlap_ratio))
        for problem in validate_dataset(questions, chunks):
            print(f"⚠️ データセット警告（chunk={chunk_size}）: {problem}")
        for strategy in args.retrievers:
            retriever = create_retriever(strategy, embedder)
            retriever.index(chunks)
            key = f"{strategy}-{chunk_size}"
            retrievers[key] = retriever
            result = evaluate_retrieval(retriever, questions, args.top_k)
            retrieval[key] = {"strategy": strategy, "chunk_size": chunk_size, "num_chunks": len(chunks), **result}
            s = result["summary"]
            print(f"[検索] {key}: Hit@1={s['hit@1']:.3f} Hit@3={s['hit@3']:.3f} MRR={s['mrr']:.3f}")

    # 2. 回答生成の評価（最初の組み合わせで、モデルごとに）
    generation: dict[str, dict] = {}
    failed_models: dict[str, str] = {}
    gen_key = f"{args.retrievers[0]}-{args.chunk_sizes[0]}"
    if not args.skip_generation:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY が設定されていません（検索だけ測る場合は --skip-generation）")
        for model in args.models:
            print(f"[生成] {model}（検索: {gen_key}）")
            try:
                generation[model] = evaluate_generation(
                    retrievers[gen_key], questions, model, args.judge_model, api_key, args.top_k, args.sleep
                )
            except Exception as e:  # モデルが混雑・利用不可でも、他のモデルの評価は続ける
                print(f"⚠️ {model} の評価を中断しました: {type(e).__name__}: {str(e)[:200]}")
                failed_models[model] = f"{type(e).__name__}: {str(e)[:200]}"

    # 3. 保存
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "questions_file": str(args.questions.relative_to(ROOT)) if args.questions.is_relative_to(ROOT) else str(args.questions),
        "pdf": str(args.pdf.relative_to(ROOT)) if args.pdf.is_relative_to(ROOT) else str(args.pdf),
        "n_questions": len(questions),
        "n_answerable": sum(1 for q in questions if q["evidence"]),
        "n_unanswerable": sum(1 for q in questions if not q["evidence"]),
        "embedding_model": "intfloat/multilingual-e5-small",
        "top_k": args.top_k,
        "gen_strategy": args.retrievers[0],
        "gen_chunk_size": args.chunk_sizes[0],
        "pricing_source": PRICING_SOURCE_URL,
        "pricing_checked": PRICING_CHECKED_DATE,
        "failed_models": failed_models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / f"{timestamp}.md"
    json_path = args.output_dir / f"{timestamp}.json"
    md_path.write_text(build_markdown(meta, retrieval, generation, args.top_k), encoding="utf-8")
    json_path.write_text(
        json.dumps({"meta": meta, "retrieval": retrieval, "generation": generation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n保存しました: {md_path}\n            {json_path}")


if __name__ == "__main__":
    main()
