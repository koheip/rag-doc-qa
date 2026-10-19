"""起動中のアプリを自動操作して、README 用のスクリーンショットを撮るスクリプト。

事前に別ターミナルで `streamlit run app.py` を起動しておきます。
    pip install playwright && playwright install chromium
    python scripts/take_screenshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "screenshot.png"
QUESTION = "有給休暇は何日前までに申請する必要がありますか？"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 画面全体が1枚に収まるよう、縦長のウィンドウにする
        page = browser.new_page(viewport={"width": 1400, "height": 2000}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        # Linux サーバーで日本語が崩れないよう、撮影時だけ日本語フォントを指定
        page.add_style_tag(content="*:not([data-testid='stIconMaterial']) { font-family: 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif !important; }")

        # 1. サンプル PDF でインデックスを作成
        page.get_by_role("button", name="インデックスを作成").click()
        page.get_by_text("チャンクを登録済み").wait_for(timeout=180_000)

        # 2. 質問を入力して送信
        chat = page.get_by_placeholder("例：有給休暇は")
        chat.fill(QUESTION)
        chat.press("Enter")
        page.get_by_text("検索されたチャンク").first.wait_for(timeout=120_000)

        # 3. 検索結果の展開欄を開いてから撮影
        page.get_by_text("検索されたチャンク").first.click()
        page.wait_for_timeout(1500)
        OUTPUT.parent.mkdir(exist_ok=True)
        page.screenshot(path=str(OUTPUT))
        browser.close()
    print(f"保存しました: {OUTPUT}")


if __name__ == "__main__":
    main()
