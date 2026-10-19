"""起動中のアプリを自動操作して、README 用のスクリーンショットを撮るスクリプト。

事前に別ターミナルで `streamlit run app.py` を起動しておきます。
（Gemini の回答を撮りたい場合は、GEMINI_API_KEY を設定した状態で起動します）
    pip install -r requirements-dev.txt && playwright install chromium
    python scripts/take_screenshot.py --output docs/screenshot.png "質問1" "質問2"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTION = "有給休暇は何日前までに申請する必要がありますか？"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*", default=[DEFAULT_QUESTION])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--output", default=str(ROOT / "docs" / "screenshot.png"))
    parser.add_argument("--height", type=int, default=2000)
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 画面全体が1枚に収まるよう、縦長のウィンドウにする
        page = browser.new_page(viewport={"width": 1400, "height": args.height}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        # Linux サーバーで日本語が崩れないよう、撮影時だけ日本語フォントを指定（アイコン用フォントは除外）
        page.add_style_tag(
            content="*:not([data-testid='stIconMaterial']) { font-family: 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif !important; }"
        )

        # 1. サンプル PDF でインデックスを作成
        page.get_by_role("button", name="インデックスを作成").click()
        page.get_by_text("チャンクを登録済み").wait_for(timeout=180_000)

        # 2. 質問を順に入力して送信し、回答（検索結果の展開欄）が出るまで待つ
        expander = page.get_by_text("検索されたチャンク")
        for n, question in enumerate(args.questions, start=1):
            chat = page.get_by_placeholder("例：有給休暇は")
            chat.fill(question)
            chat.press("Enter")
            expander.nth(n - 1).wait_for(timeout=120_000)

        # 3. 最初の回答の検索結果を展開してから撮影
        expander.first.click()
        page.wait_for_timeout(1500)
        Path(args.output).parent.mkdir(exist_ok=True)
        page.screenshot(path=args.output)
        browser.close()
    print(f"保存しました: {args.output}")


if __name__ == "__main__":
    main()
