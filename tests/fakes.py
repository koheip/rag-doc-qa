"""テスト用の軽量な偽物（Fake）クラス。本物のモデルや API を使わずに高速にテストするためのもの。"""

import numpy as np

from src.llm_client import LLMResponse


class FakeEmbedder:
    """各文字をハッシュで次元に割り当て、出現回数を数えるだけの埋め込み。

    同じ文字を多く含む文章ほど類似度が高くなるので、検索の動作確認には十分です。
    """

    dim = 256

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for ch in text:
            vec[ord(ch) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts):
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text):
        return self._embed(text)


class FakeLLM:
    """決まった文字列を返すだけの LLM クライアント（受け取ったプロンプトは記録する）。"""

    model = "fake-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, system_prompt="", temperature=0.1, json_mode=False):
        self.prompts.append(prompt)
        return LLMResponse(text=self.reply, model=self.model, input_tokens=100, output_tokens=10, latency_s=0.01)
