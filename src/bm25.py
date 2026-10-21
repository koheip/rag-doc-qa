"""BM25（キーワード検索）と、密ベクトル検索を組み合わせた「ハイブリッド検索」のモジュール。

==============================================================================
# TODO(弘平さんの課題): このファイルの NotImplementedError の部分を実装してください。
==============================================================================

■ なぜハイブリッド検索？
  密ベクトル検索（DenseRetriever）は「意味」で探すので言い換えに強い一方、
  「第21条」「内線1234」「14,000円」のような **文字そのものの一致** が大事な質問は苦手です。
  BM25 は単語（トークン）の一致で探すので、その弱点を補えます。
  2つの検索結果を「順位」で統合（RRF）すると、両方の長所を活かせます。

■ 実装する順番（おすすめ）
  1. tokenize()            … 日本語テキストをトークンに分ける
  2. BM25Index.fit/scores  … BM25 スコアを計算する
  3. reciprocal_rank_fusion() … 2つのランキングを統合する
  4. HybridRetriever.index/retrieve … 上の部品を組み合わせる

■ 確認方法
  python -m pytest -m exercise -v        # 課題用テスト（通常の pytest では実行されません）
  python eval/run_eval.py --retrievers dense hybrid --skip-generation   # 精度がどれだけ上がったか測る

■ ヒント
  - 日本語の分かち書き（MeCab など）は不要です。「文字 n-gram」で十分効果があります。
    例: "給与の支払い" を 2-gram にすると ["給与", "与の", "の支", "支払", "払い"]
  - BM25 の式（1つのクエリトークン t について。合計したものが文書のスコア）:
        idf(t)   = log( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
        score    = idf(t) * tf * (k1 + 1) / ( tf + k1 * (1 - b + b * 文書長 / 平均文書長) )
      N: 文書数, n_t: t を含む文書数, tf: 文書内での t の出現回数
  - RRF（Reciprocal Rank Fusion）: 各ランキングでの順位 r（1始まり）から 1 / (k + r) を足し合わせる。
      k=60 がよく使われる値。スコアの尺度が違う検索結果でも、順位だけで公平に統合できるのが利点。
  - numpy は使っても使わなくても OK。外部ライブラリ（rank-bm25 など）は使わずに書いてみましょう。

■ 発展課題（テストが通った後に）
  ハイブリッドにすれば必ず良くなるとは限りません。評価スクリプトで dense と比べてみて、
  もし悪くなっていたら「なぜか」を、質問タイプ別の結果や外れた質問から考えてみましょう。
  例えば、2-gram には「です」「ます」「の支」のような意味の薄いトークンも大量に含まれます。
  改善のアイデア: ひらがなだけのトークンを BM25 で無視する / RRF で BM25 側の重みを下げる（重み付き RRF）など。
  改善したら、必ず評価の数字で効果を確かめること（Hit@1・Hit@3・MRR がどう変わったか）。
"""

from __future__ import annotations

from src.chunker import Chunk
from src.retriever import DenseRetriever, RetrievedChunk


def tokenize(text: str, n: int = 2) -> list[str]:
    """テキストを文字 n-gram のトークン列に変換する。

    仕様:
      - 空白（スペース・改行）と句読点・記号は事前に取り除く（例: 「、」「。」「（」「）」「・」など）
      - 英字は小文字にそろえる
      - 残った文字列の長さが n 未満なら、その文字列1つだけをトークンとする（空なら空リスト）

    例:
        tokenize("給与の支払い") == ["給与", "与の", "の支", "支払", "払い"]
        tokenize("PC") == ["pc"]
    """
    # TODO(弘平さんの課題)
    raise NotImplementedError


class BM25Index:
    """BM25 によるキーワード検索のインデックス。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # k1: 同じ単語が何度も出たときに、スコアをどこまで増やすか（大きいほど回数を重視）
        # b : 文書の長さでどれだけ補正するか（0=補正なし, 1=完全に補正）
        self.k1 = k1
        self.b = b
        # TODO(弘平さんの課題): fit() で計算した値を保存する変数を用意する
        #   例: 各文書のトークン出現回数（collections.Counter のリスト）、文書長、平均文書長、各トークンの文書頻度

    def fit(self, texts: list[str]) -> None:
        """文書（チャンクの本文）のリストから、BM25 の計算に必要な統計を作る。"""
        # TODO(弘平さんの課題)
        raise NotImplementedError

    def scores(self, query: str) -> list[float]:
        """クエリに対する各文書の BM25 スコアを、fit() に渡した順番で返す。

        クエリのトークンが1つも文書に無ければ、全文書 0.0 になる。
        """
        # TODO(弘平さんの課題)
        raise NotImplementedError


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """複数のランキング（文書番号を良い順に並べたリスト）を RRF で1つに統合する。

    Args:
        rankings: 例 [[2, 0, 1], [0, 2, 3]]（1つ目の検索では文書2が1位、2つ目では文書0が1位）
        k: RRF の定数（大きいほど上位と下位の差が小さくなる）

    Returns:
        (文書番号, RRF スコア) のリスト。スコアの高い順。
        どれか1つのランキングにでも登場した文書はすべて含める。
    """
    # TODO(弘平さんの課題)
    raise NotImplementedError


class HybridRetriever:
    """BM25 と密ベクトル検索の結果を RRF で統合する検索器（Retriever プロトコルを満たす）。"""

    def __init__(self, dense: DenseRetriever, candidates: int = 20, rrf_k: int = 60) -> None:
        # candidates: 統合する前に、それぞれの検索で上位何件まで候補を取るか
        self.dense = dense
        self.bm25 = BM25Index()
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """密ベクトル側と BM25 側の両方にチャンクを登録する。

        空のリストなら ValueError（DenseRetriever と同じ挙動）。
        """
        # TODO(弘平さんの課題)
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """BM25 と密ベクトル検索をそれぞれ実行し、RRF で統合した上位 top_k 件を返す。

        ヒント:
          - 密ベクトル側: self.dense.retrieve(query, self.candidates) の結果は Chunk なので、
            index() のときに「chunk_id → self.chunks の何番目か」の辞書を作っておくと文書番号に変換できる
          - BM25 側: self.bm25.scores(query) をスコアの高い順に並べ、スコア 0 の文書は除く
          - 返す RetrievedChunk の score には RRF スコアを入れる
          - index() 前に呼ばれたら RuntimeError
        """
        # TODO(弘平さんの課題)
        raise NotImplementedError
