"""LLM（大規模言語モデル）を呼び出す部分を、提供元ごとに差し替えられるようにするモジュール。

  - LLMClient    : LLM クライアントが満たすべきインターフェース（Protocol）
  - LLMResponse  : 回答テキストに加え、トークン数・応答時間を持つ結果
  - GeminiClient : Google Gemini（公式 SDK google-genai）による実装

Azure OpenAI や Amazon Bedrock に対応したい場合は、同じ generate() を持つクラスを
追加するだけで、プロンプトやアプリ側のコードは変えずに済みます。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """LLM の応答。評価（費用・速度の計測）にも使えるよう、使用量も記録する。"""

    text: str
    model: str
    input_tokens: int = 0  # 入力（プロンプト）のトークン数
    output_tokens: int = 0  # 出力のトークン数（思考トークンも含む。課金対象のため）
    latency_s: float = 0.0  # 呼び出しにかかった秒数（リトライの待ち時間は含まない）


class LLMClient(Protocol):
    """LLM クライアントのインターフェース。"""

    model: str

    def generate(
        self, prompt: str, system_prompt: str = "", temperature: float = 0.1, json_mode: bool = False
    ) -> LLMResponse:
        """プロンプトを送り、応答を返す。json_mode=True なら JSON 形式で出力させる。"""
        ...


class GeminiClient:
    """Google Gemini を呼び出すクライアント。

    無料枠ではリクエスト数の制限（レート制限）があるので、
    429（回数超過）や 5xx（一時的なサーバーエラー）のときは、待ち時間を倍々に延ばして再試行します（指数バックオフ）。
    """

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    MAX_SERVER_SUGGESTED_WAIT_S = 120  # サーバーが「これ以上待て」と言う時間がこれより長ければ再試行しない

    def __init__(
        self, api_key: str, model: str, max_retries: int = 3, initial_wait_s: float = 5.0, timeout_s: float = 60.0
    ) -> None:
        # google-genai は公式の新しい SDK（旧 google-generativeai の後継）
        from google import genai
        from google.genai import types

        # timeout を付けないと、サーバーが混雑しているときに応答を待ち続けて固まることがある
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.model = model
        self.max_retries = max_retries
        self.initial_wait_s = initial_wait_s

    def generate(
        self, prompt: str, system_prompt: str = "", temperature: float = 0.1, json_mode: bool = False
    ) -> LLMResponse:
        import httpx
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        wait = self.initial_wait_s
        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                latency = time.perf_counter() - start
                break
            except (errors.APIError, httpx.TimeoutException) as e:
                code = getattr(e, "code", "timeout")  # タイムアウトは HTTP ステータスが無い
                if (code != "timeout" and code not in self.RETRYABLE_STATUS) or attempt == self.max_retries:
                    raise
                # 429 のとき、サーバーが待つべき秒数（RetryInfo）を教えてくれることがある。
                # 1日あたりの上限に達した場合は「約24時間後」と返ってくるので、待たずにあきらめる。
                suggested = _suggested_retry_delay_s(e)
                if suggested is not None and suggested > self.MAX_SERVER_SUGGESTED_WAIT_S:
                    raise
                sleep_s = max(wait, suggested or 0)
                print(f"  ({self.model}: {code} のため {sleep_s:.0f} 秒待って再試行します)", file=sys.stderr, flush=True)
                time.sleep(sleep_s)
                wait *= 2

        usage = response.usage_metadata
        return LLMResponse(
            text=(response.text or "").strip(),
            model=self.model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            latency_s=latency,
        )


def _suggested_retry_delay_s(error: Exception) -> float | None:
    """API エラーの詳細から、サーバーが勧める待ち時間（秒）を取り出す。無ければ None。"""
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None
    for item in details.get("error", {}).get("details", []):
        if item.get("@type", "").endswith("RetryInfo"):
            try:
                return float(str(item.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                return None
    return None
