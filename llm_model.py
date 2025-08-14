
import asyncio
from langchain import requests
from langchain_core.language_models import LLM
from typing import Optional, List
import g4f
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)



class CustomLLM(LLM):
    api_token: str
    model_name: str
    base_url: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                # OpenAI ChatCompletion yanıtı
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "Yanıt alınamadı."
            else:
                return f"Hata: {response.status_code} - {response.text}"
        except Exception as e:
            return f"API çağrısında hata oluştu: {e}"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}




warnings.filterwarnings("ignore")



class Gpt4FreeLLM(LLM):
    g4f.debug.logging = False

    @property
    def _llm_type(self) -> str:
        return "gpt4free_llm"

    async def _try_model(self, model: str, prompt: str) -> Optional[str]:
        """Tek bir modeli async olarak dener"""
        try:
            # asyncio.to_thread ile bloklayan kodu paralel çalıştırıyoruz
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    g4f.ChatCompletion.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    web_search=False,
                    timeout=5  # modelin kendi timeout'u
                ),
                timeout=7  # asyncio seviyesinde ek güvenlik
            )
            if response and len(str(response)) > 10 and "Login" not in str(response):
                return str(response)
        except Exception:
            return None
        return None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Paralel olarak tüm modelleri dener ve ilk gelen cevabı döner"""
        models = [
            # Hızlı ve stabil olanlar
            "gpt-4o-mini",
            "claude-3-haiku",
            "mistral-7b",
            "deepseek-v3",
            "gemini-pro",
            "gpt-3.5-turbo",
            # İkinci grup
            "gpt-4-turbo",
            "gpt-4",
            "gemini-1.5-pro",
            "claude-3-sonnet",
        ]

        async def run_all():
            tasks = [self._try_model(m, prompt) for m in models]
            # İlk dönen sonucu almak için as_completed kullanıyoruz
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    return result
            return None

        # Async kodu sync içinde çalıştırma
        result = asyncio.run(run_all())

        if result:
            return result
        else:
            return "AI servisi şu anda yanıt vermiyor. Lütfen tekrar deneyin."

    @property
    def _identifying_params(self) -> dict:
        return {}
