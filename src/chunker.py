"""ページのテキストを、検索しやすい長さの「チャンク」に分割するモジュール。

日本語は英語と違って単語の間にスペースが無いため、「単語数」ではなく
「文字数」で長さを測り、句点（。）などの文の区切りを優先して分割します。
また、チャンクの境界で文脈が途切れないよう、前のチャンクの末尾を
少し重ねる（オーバーラップ）ようにしています。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.loader import Page

# 「。」「！」「？」や改行の直後で文を区切るための正規表現
# （「。」の直後に改行が続く場合は、改行まで含めて1文とする）
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?\n])(?!\n)")


@dataclass
class Chunk:
    """検索の単位となるテキストの断片。出典情報（ファイル名・ページ）を持つ。"""

    text: str
    source: str
    page: int
    chunk_id: int  # 全チャンクの通し番号


def split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する（区切り文字・改行は文の末尾に残す）。"""
    return [s.lstrip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def split_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """テキストを chunk_size 文字以下のチャンクに分割する。

    手順:
      1. 文単位に分ける
      2. chunk_size を超えない範囲で文を順に詰めていく
      3. 新しいチャンクを始めるとき、直前チャンクの末尾 overlap 文字ぶんの文を引き継ぐ
      4. 1文だけで chunk_size を超える長文は、文字数で強制的に切る
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size は 1 以上にしてください")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap は 0 以上かつ chunk_size 未満にしてください")

    # 長すぎる文は先に分割しておく（スライディングウィンドウ方式）
    sentences: list[str] = []
    step = chunk_size - overlap
    for sentence in split_sentences(text):
        if len(sentence) <= chunk_size:
            sentences.append(sentence)
        else:
            sentences.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), step))

    chunks: list[str] = []
    current: list[str] = []  # 今作っているチャンクに入っている文
    for sentence in sentences:
        if current and len("".join(current)) + len(sentence) > chunk_size:
            chunks.append("".join(current).strip())
            # オーバーラップ: 末尾から overlap 文字に収まるだけ文を引き継ぐ
            carried: list[str] = []
            for prev in reversed(current):
                if len("".join(carried)) + len(prev) > overlap:
                    break
                carried.insert(0, prev)
            # 引き継いだ文と次の文で chunk_size を超えるなら引き継ぎをやめる
            if len("".join(carried)) + len(sentence) > chunk_size:
                carried = []
            current = carried
        current.append(sentence)
    if current:
        chunks.append("".join(current).strip())
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """ページのリストをチャンクのリストに変換する。

    ページをまたいでチャンクを作らないことで、出典のページ番号を正確に保てます。
    """
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=text, source=page.source, page=page.page, chunk_id=len(chunks)))
    return chunks
