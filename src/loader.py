"""PDF を読み込み、ページ単位のテキストに変換するモジュール。

ページ番号を保持しておくことで、回答に「どのファイルの何ページか」という
出典（引用）を付けられるようにします。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

# 文の終わりとみなす文字。この直後の改行は「本当の改行」として残す。
_SENTENCE_END = "。！？!?」）)"

# 「- 2 / 6 -」「3」のような、ページ番号だけの行（ヘッダー・フッター）
_PAGE_NUMBER_LINE = re.compile(r"^[-\s]*\d+(\s*/\s*\d+)?[-\s]*$")


@dataclass
class Page:
    """PDF の1ページ分のテキスト。"""

    source: str  # ファイル名（出典表示に使う）
    page: int  # ページ番号（1始まり）
    text: str


def clean_text(text: str) -> str:
    """PDF 抽出テキストを整形する。

    PDF では見た目の折り返し位置で改行が入るため、日本語の文の途中に
    不要な改行が混ざります。文末以外の改行は取り除いて文をつなげます。
    """
    lines = [line.strip() for line in text.splitlines()]
    # 空行とページ番号だけの行は検索の邪魔になるので捨てる
    lines = [line for line in lines if line and not _PAGE_NUMBER_LINE.match(line)]
    if not lines:
        return ""

    result = lines[0]
    for line in lines[1:]:
        prev_char, next_char = result[-1], line[0]
        if prev_char in _SENTENCE_END:
            result += "\n" + line  # 文の終わり → 改行を残す
        elif prev_char.isascii() and next_char.isascii() and prev_char.isalnum():
            result += " " + line  # 英単語どうしはスペースでつなぐ
        else:
            result += line  # 日本語の途中 → そのままつなぐ
    # 連続する空白を1つにまとめる
    return re.sub(r"[ \t\u3000]+", " ", result).strip()


def load_pdf(file: str | Path | BinaryIO | bytes, source_name: str | None = None) -> list[Page]:
    """PDF を読み込み、ページごとの Page のリストを返す。

    Args:
        file: ファイルパス、ファイルオブジェクト（Streamlit のアップロードなど）、またはバイト列。
        source_name: 出典として表示する名前。省略時はファイル名を使う。
    """
    if isinstance(file, bytes):
        file = io.BytesIO(file)
    if source_name is None:
        source_name = Path(file).name if isinstance(file, (str, Path)) else getattr(file, "name", "document.pdf")

    reader = PdfReader(file)
    pages: list[Page] = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text:  # 画像だけのページなど、文字が取れないページは飛ばす
            pages.append(Page(source=source_name, page=i, text=text))
    return pages
