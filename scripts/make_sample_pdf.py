"""サンプルPDF（架空の会社の就業規則・社内FAQ）を生成するスクリプト。

実在の企業・人物とは一切関係のない架空の内容なので、公開しても問題ありません。
reportlab と日本語TrueTypeフォントを使って、複数ページのPDFを作ります。

使い方:
    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "sample_docs" / "sample_company_rules.pdf"

# 日本語フォントの候補（環境によって入っているものが違うので順に探す）
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/sand-box/google/BIZ UDPMincho/BIZUDPMincho-Regular.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]

# ページごとの内容: (見出し, [段落, ...])
PAGES: list[tuple[str, list[str]]] = [
    (
        "株式会社サンプルテック 就業規則（抜粋）",
        [
            "本書は架空の企業「株式会社サンプルテック」の就業規則および社内FAQをまとめたデモ用資料です。実在の企業・団体とは関係ありません。",
            "第1条（目的）この規則は、株式会社サンプルテック（以下「会社」という）の従業員の労働条件、服務規律その他の就業に関する事項を定めるものである。",
            "第2条（勤務時間）所定労働時間は1日8時間とし、始業は午前9時、終業は午後6時とする。休憩時間は正午から午後1時までの60分間とする。",
            "第3条（フレックスタイム制）会社はフレックスタイム制を導入する。コアタイムは午前11時から午後3時までとし、それ以外の時間帯は従業員が始業・終業時刻を自由に決めることができる。",
            "第4条（リモートワーク）従業員は上長の承認を得たうえで、週3日までリモートワークを行うことができる。リモートワーク手当として月額5,000円を支給する。",
        ],
    ),
    (
        "第2章 休暇",
        [
            "第5条（年次有給休暇）入社日から6か月間継続勤務した従業員には、10日の年次有給休暇を付与する。以後、勤続年数に応じて付与日数を加算し、最大20日とする。",
            "第6条（有給休暇の申請）年次有給休暇を取得する場合は、原則として取得希望日の3営業日前までに勤怠システムから申請しなければならない。急病などやむを得ない場合は当日の申請も認める。",
            "第7条（特別休暇）従業員本人が結婚するときは5日、配偶者が出産するときは3日、親族が亡くなったときは最大5日の特別休暇（有給）を与える。",
            "第8条（リフレッシュ休暇）勤続5年ごとに、連続5日間のリフレッシュ休暇と旅行補助金3万円を支給する。",
        ],
    ),
    (
        "第3章 給与・経費",
        [
            "第9条（給与の支払い）給与は毎月末日締めとし、翌月25日に従業員指定の銀行口座へ振り込む。25日が銀行休業日の場合はその前営業日に支払う。",
            "第10条（通勤手当）通勤手当は、最も経済的かつ合理的な経路による実費を支給する。ただし上限は月額3万円とする。",
            "第11条（経費精算）業務で立て替えた経費は、領収書を添えて発生月の翌月10日までに経費精算システムで申請する。1件5万円を超える支出は事前に部門長の承認が必要である。",
            "第12条（書籍・研修補助）業務に関連する書籍購入やオンライン講座の受講費用は、年間10万円まで会社が負担する。",
        ],
    ),
    (
        "第4章 服務規律・情報セキュリティ",
        [
            "第13条（服務の基本）従業員は、会社の規則を守り、互いの人格を尊重して誠実に職務を遂行しなければならない。",
            "第14条（情報の持ち出し）顧客情報や社外秘の資料を、会社の許可なく社外へ持ち出したり、個人のクラウドストレージに保存したりしてはならない。",
            "第15条（パスワード管理）業務システムのパスワードは12文字以上とし、他人と共有してはならない。パスワードを紛失した場合は、速やかに情報システム部へ届け出ること。",
            "第16条（生成AIの利用）業務で生成AIツールを利用する場合は、会社が承認したツールのみを使用し、顧客の個人情報や機密情報を入力してはならない。",
            "第17条（セキュリティ事故の報告）ウイルス感染やPCの紛失・盗難などのセキュリティ事故が発生した場合は、直ちに上長と情報システム部に報告しなければならない。",
        ],
    ),
    (
        "第5章 育児・介護・退職",
        [
            "第18条（育児休業）従業員は、子が1歳6か月になるまで育児休業を取得できる。男性従業員の取得も推奨する。",
            "第19条（介護休業）家族の介護が必要な従業員は、対象家族1人につき通算93日まで介護休業を取得できる。",
            "第20条（時短勤務）小学校3年生までの子を養育する従業員は、1日6時間の短時間勤務を選択できる。",
            "第21条（退職の手続き）自己都合で退職する場合は、退職希望日の1か月前までに退職届を上長に提出しなければならない。",
            "第22条（貸与品の返却）退職時には、社員証、PC、その他会社から貸与された物品をすべて返却しなければならない。",
        ],
    ),
    (
        "社内FAQ（よくある質問）その1",
        [
            "Q. パソコンが故障したときはどうすればよいですか。A. 情報システム部のヘルプデスク（内線1234）に連絡してください。代替機は原則として当日中に貸し出します。",
            "Q. 社内Wi-Fiのパスワードはどこで確認できますか。A. セキュリティ上の理由からPDFには記載していません。社内ポータルの「IT案内」ページを確認してください。",
            "Q. 健康診断はいつ受けられますか。A. 毎年10月に全従業員を対象とした定期健康診断を実施します。受診費用は会社が全額負担します。",
            "Q. 副業は認められていますか。A. 事前に副業届を人事部へ提出し、承認を得た場合に限り認められます。競合企業での就業は認められません。",
        ],
    ),
    (
        "社内FAQ（よくある質問）その2",
        [
            "Q. 名刺はどうやって注文しますか。A. 総務部の申請フォームから注文してください。通常5営業日で届きます。",
            "Q. 会議室の予約方法を教えてください。A. グループウェアの施設予約から予約できます。1回の予約は最大2時間までです。",
            "Q. 引っ越しをしたらどんな手続きが必要ですか。A. 引っ越し後2週間以内に、人事システムで住所変更と通勤経路の変更を申請してください。通勤手当は申請内容に基づいて再計算されます。",
            "Q. 社内の勉強会に参加できますか。A. 毎週金曜日の午後5時から、エンジニア有志による技術勉強会を開催しています。どなたでも参加できます。",
        ],
    ),
]


def find_font() -> str:
    """利用可能な日本語フォントを登録し、フォント名を返す。"""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("JPFont", path))
            return "JPFont"
    # どれも無い場合は reportlab 内蔵の日本語CIDフォントを使う
    print("日本語TTFが見つからないため内蔵CIDフォントを使います", file=sys.stderr)
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    return "HeiseiKakuGo-W5"


def wrap_japanese(text: str, font: str, size: float, max_width: float) -> list[str]:
    """日本語は単語の区切りが無いので、1文字ずつ幅を測って折り返す。"""
    lines, current = [], ""
    for ch in text:
        if pdfmetrics.stringWidth(current + ch, font, size) > max_width:
            lines.append(current)
            current = ch
        else:
            current += ch
    if current:
        lines.append(current)
    return lines


def main() -> None:
    font = find_font()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=A4)
    c.setTitle("株式会社サンプルテック 就業規則・社内FAQ（架空）")
    width, height = A4
    margin = 20 * mm
    max_width = width - margin * 2

    for page_no, (title, paragraphs) in enumerate(PAGES, start=1):
        y = height - margin
        c.setFont(font, 16)
        c.drawString(margin, y, title)
        y -= 14 * mm
        c.setFont(font, 10.5)
        for para in paragraphs:
            for line in wrap_japanese(para, font, 10.5, max_width):
                c.drawString(margin, y, line)
                y -= 6 * mm
            y -= 4 * mm  # 段落の間を空ける
        # フッター（ページ番号）
        c.setFont(font, 9)
        c.drawString(width / 2 - 10 * mm, 12 * mm, f"- {page_no} / {len(PAGES)} -")
        c.showPage()

    c.save()
    print(f"サンプルPDFを作成しました: {OUTPUT}")


if __name__ == "__main__":
    main()
