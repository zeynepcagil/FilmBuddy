
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

import asyncio
from typing import Optional, List
import g4f
from langchain_core.language_models import LLM
import tiktoken
from counter import llm_counter

# Token sayımı için bir yardımcı fonksiyon
def get_token_count(text: str, model_name: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Model bulunamazsa genel bir encodig kullan
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class Gpt4FreeLLM(LLM):
    g4f.debug.logging = False

    @property
    def _llm_type(self) -> str:
        return "gpt4free_llm"

    async def _try_model(self, model: str, prompt: str) -> Optional[str]:
        """Tek bir modeli async olarak dener ve başarılı olursa tokenları sayar"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    g4f.ChatCompletion.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    web_search=False,
                    timeout=5
                ),
                timeout=7
            )
            if response and len(str(response)) > 10 and "Login" not in str(response):
                # Başarılı yanıt gelince sayaçları güncelle
                input_tokens = get_token_count(prompt, model)
                output_tokens = get_token_count(str(response), model)
                llm_counter.increment_call()
                llm_counter.update_tokens(input_tokens, output_tokens)
                return str(response)
        except Exception:
            return None
        return None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Paralel olarak tüm modelleri dener ve ilk gelen cevabı döner"""
        models = [
            "gpt-4o-mini", "claude-3-haiku", "mistral-7b", "deepseek-v3",
            "gemini-pro", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4",
            "gemini-1.5-pro", "claude-3-sonnet", "mistral-instruct",
            "gemini-2", "gpt-neo-2.7b",
        ]

        async def run_all():
            tasks = [self._try_model(m, prompt) for m in models]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    return result
            return None

        result = asyncio.run(run_all())

        if result:
            return result
        else:
            return "AI servisi şu anda yanıt vermiyor. Lütfen tekrar deneyin."

    @property
    def _identifying_params(self) -> dict:
        return {}