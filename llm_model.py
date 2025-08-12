import warnings

from langchain import requests
import warnings
from typing import Optional, List
import g4f
from langchain_core.language_models.llms import LLM


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

        # OpenAI Chat Completion için mesaj formatı
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
    """
    A custom LangChain LLM that uses the g4f library
    to access various free large language models.
    """

    @property
    def _llm_type(self) -> str:
        """Returns the type of the LLM."""
        return "gpt4free_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        The core method that makes a call to the underlying g4f models.
        """
        models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o-mini",
            "claude-3-sonnet",
            "gemini-pro",
            "deepseek-chat",
            "mixtral-8x7b",
            "deepseek-r1",
            "gemini-1.5-pro",
            "gpt-4.1-mini"

        ]

        # Method 1: Try different models sequentially
        for model in models:
            try:
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    web_search=False
                )
                if response and len(str(response)) > 10 and "Login" not in str(response):
                    return str(response)
            except Exception:
                continue

        # Add your other fallback methods here if you want them.
        # This part of the code remains the same as in your original file.

        return "AI servisi geçici olarak kullanılamıyor. Lütfen daha sonra tekrar deneyin."

    @property
    def _identifying_params(self) -> dict:
        """Returns a dictionary of identifying parameters."""
        return {}