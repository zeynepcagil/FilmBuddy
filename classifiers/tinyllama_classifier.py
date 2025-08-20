from transformers import pipeline

class LlamaClassifier:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.pipe = pipeline("text-generation", model=model_name)

    def classify(self, text: str):
        prompt = f"""
Kullanıcı mesajını kategorize et:
Mesaj: "{text}"

Kategoriler:
- chat
- recommendation
- other

Sadece yukarıdaki kategorilerden birini yaz:
"""
        result = self.pipe(
            prompt,
            max_new_tokens=20,   # Sadece üretilecek token sayısı
            do_sample=False,
            truncation=True      # Giriş uzunluğunu kes
        )[0]["generated_text"]

        # Çıktıdan ilk geçerli kategoriyi ayıkla
        for cat in ["chat", "recommendation", "other"]:
            if cat in result.lower():
                return cat
        return "other"


class LlamaChat:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.pipe = pipeline("text-generation", model=model_name)

    def chat(self, text: str):
        prompt = f"Kullanıcı: {text}\nAsistan:"
        response = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            truncation=True
        )[0]["generated_text"]
        # Yanıtı kullanıcı kısmından sonra döndür
        return response.split("Asistan:")[-1].strip()
