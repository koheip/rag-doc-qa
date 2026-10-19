"""PDF ドキュメント Q&A（RAG デモ）の Streamlit アプリ本体。

起動方法:
    streamlit run app.py

画面の流れ:
    1. サイドバーで設定（API キー・チャンクサイズ・top-k など）
    2. PDF をアップロード（またはサンプル PDF を使用）して「インデックス作成」
    3. チャット欄に質問 → 回答と出典、検索されたチャンクが表示される
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_pages
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODELS,
)
from src.embedder import E5Embedder
from src.llm import cited_indices, extractive_answer, format_citation, generate_answer
from src.loader import load_pdf
from src.retriever import RetrievedChunk, Retriever

# .env ファイルがあれば、その中身を環境変数として読み込む（GEMINI_API_KEY など）
load_dotenv()

SAMPLE_PDF = Path(__file__).parent / "sample_docs" / "sample_company_rules.pdf"

st.set_page_config(page_title="PDF ドキュメント Q&A（RAG）", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# 重い処理はキャッシュして、画面操作のたびに再実行されないようにする
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="埋め込みモデルを読み込み中です（初回のみ数十秒かかります）...")
def get_embedder() -> E5Embedder:
    return E5Embedder(EMBEDDING_MODEL_NAME)


def get_api_key(sidebar_input: str) -> str | None:
    """API キーを 環境変数 → st.secrets → サイドバー入力 の順で探す。

    キーはどこにも保存せず、このリクエストの処理中だけ使います。
    """
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # secrets.toml が無い場合は例外になるので無視する
        pass
    return sidebar_input or None


def build_index(files: list[tuple[str, bytes]], chunk_size: int, overlap: int) -> Retriever:
    """PDF 群を読み込み → チャンク分割 → ベクトル化して検索器を作る。"""
    pages = []
    for name, data in files:
        pages.extend(load_pdf(data, source_name=name))
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    retriever = Retriever(get_embedder())
    retriever.index(chunks)
    return retriever


def render_sources(contexts: list[RetrievedChunk], cited: list[int]) -> None:
    """出典（回答が実際に参照した資料）と、検索されたチャンクの中身（展開表示）を描画する。

    cited は回答中の参照番号（[1] なら 1）。検索しただけで回答に使われなかった資料は
    出典には出さず、展開表示の中でだけ確認できるようにする。
    """
    if not contexts:
        return
    if cited:
        st.caption("📎 出典: " + " / ".join(f"[{i}] {format_citation(contexts[i - 1])}" for i in cited))
    else:
        st.caption("📎 出典: なし（資料から該当する記述が見つかりませんでした）")
    with st.expander(f"🔍 検索されたチャンク（{len(contexts)} 件）を見る"):
        for i, item in enumerate(contexts, start=1):
            st.markdown(f"**[{i}] {format_citation(item)}** — 類似度 `{item.score:.3f}`")
            st.progress(max(0.0, min(1.0, item.score)))
            st.text(item.chunk.text)


# ---------------------------------------------------------------------------
# セッション状態（ページを再描画しても消えないデータ）の初期化
# ---------------------------------------------------------------------------
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("messages", [])  # チャット履歴
st.session_state.setdefault("indexed_files", [])

# ---------------------------------------------------------------------------
# サイドバー: 設定
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("Gemini API")
    key_input = st.text_input(
        "API キー（任意）",
        type="password",
        help="入力しない場合は、検索結果をそのまま表示する『抽出モード』で動作します。キーは保存されません。",
    )
    api_key = get_api_key(key_input)
    model_name = st.selectbox("モデル", GEMINI_MODELS, index=0)
    if api_key:
        st.success("生成モード：Gemini で回答を作成します")
    else:
        st.info("抽出モード：API キー無しで、関連箇所を表示します（無料）")

    st.subheader("検索パラメータ")
    chunk_size = st.slider("チャンクサイズ（文字数）", 100, 1000, DEFAULT_CHUNK_SIZE, step=50)
    overlap = st.slider("オーバーラップ（文字数）", 0, chunk_size // 2, min(DEFAULT_CHUNK_OVERLAP, chunk_size // 2), step=10)
    top_k = st.slider("取得するチャンク数（top-k）", 1, 10, DEFAULT_TOP_K)
    st.caption("※ チャンクサイズ・オーバーラップを変えたら、インデックスを作り直してください。")

    if st.button("🗑️ 会話をリセット", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# メイン画面: ドキュメントの登録
# ---------------------------------------------------------------------------
st.title("📄 PDF ドキュメント Q&A")
st.write(
    "PDF をアップロードして質問すると、関連する箇所を検索し、その内容を根拠に回答します"
    "（RAG: Retrieval-Augmented Generation）。"
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded_files = st.file_uploader("PDF を選択（複数可）", type=["pdf"], accept_multiple_files=True)
with col_sample:
    st.write("")
    use_sample = st.checkbox("サンプル PDF を使う", value=not uploaded_files, help="架空の会社の就業規則・社内FAQです")

if st.button("📚 インデックスを作成", type="primary"):
    files = [(f.name, f.getvalue()) for f in uploaded_files or []]
    if use_sample and SAMPLE_PDF.exists():
        files.append((SAMPLE_PDF.name, SAMPLE_PDF.read_bytes()))
    if not files:
        st.warning("PDF をアップロードするか、「サンプル PDF を使う」にチェックを入れてください。")
    else:
        try:
            with st.spinner("PDF を読み込み、ベクトル化しています..."):
                st.session_state.retriever = build_index(files, chunk_size, overlap)
            st.session_state.indexed_files = [name for name, _ in files]
            st.session_state.messages = []
        except ValueError:
            st.error("PDF からテキストを取り出せませんでした（画像だけの PDF は未対応です）。")

retriever: Retriever | None = st.session_state.retriever
if retriever:
    st.success(
        f"✅ {len(st.session_state.indexed_files)} ファイル / {retriever.num_chunks} チャンクを登録済み："
        + "、".join(st.session_state.indexed_files)
    )
else:
    st.info("まず PDF を登録してください（サンプル PDF ならそのまま「インデックスを作成」を押すだけです）。")

st.divider()

# ---------------------------------------------------------------------------
# メイン画面: チャット
# ---------------------------------------------------------------------------
# これまでの会話を表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("contexts", []), msg.get("cited", []))

placeholder = "例：有給休暇は何日前までに申請すればいいですか？" if retriever else "先にインデックスを作成してください"
if question := st.chat_input(placeholder, disabled=retriever is None):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1. 検索（Retrieval）
        with st.spinner("関連する箇所を検索中..."):
            contexts = retriever.retrieve(question, top_k=top_k)

        # 2. 生成（Generation）。キーが無ければ抽出モード
        # 抽出モードでは、表示した上位2件をそのまま出典とする
        cited = list(range(1, min(2, len(contexts)) + 1))
        if api_key:
            try:
                with st.spinner(f"{model_name} で回答を生成中..."):
                    answer = generate_answer(question, contexts, api_key=api_key, model=model_name)
                cited = cited_indices(answer, len(contexts))
            except Exception as e:  # キー間違い・レート制限など
                answer = (
                    f"⚠️ Gemini の呼び出しに失敗したため、抽出モードで表示します。（{type(e).__name__}）\n\n"
                    + extractive_answer(contexts)
                )
        else:
            answer = (
                "ℹ️ API キーが未設定のため、**抽出モード**（検索結果をそのまま表示）で回答しています。\n\n"
                + extractive_answer(contexts)
            )

        st.markdown(answer)
        render_sources(contexts, cited)

    st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts, "cited": cited})
