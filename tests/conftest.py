"""テスト共通の設定。プロジェクト直下を import パスに追加し、`from src...` で読めるようにする。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
