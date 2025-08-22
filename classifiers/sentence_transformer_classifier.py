
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict

class SentenceTransformerClassifier:
    """
    Bu sınıf, bir metni, önceden belirlenmiş etiketlere göre sınıflandırmak için
    Sentence-BERT modelini kullanır.
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Sınıflandırma modelini başlatır.

        Args:
            model_name (str): Yüklenecek Sentence Transformer modelinin adı.
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Sentence Transformer modeli '{model_name}' başarıyla yüklendi.")
        except Exception as e:
            print(f"Hata: Sentence Transformer modeli yüklenirken bir sorun oluştu: {e}")
            self.model = None

        self.labels = []
        self.label_embeddings = None

    def set_labels(self, labels: List[str]):
        """
        Sınıflandırma için kullanılacak etiketleri (kategorileri) belirler.
        Etiketlerin vektörel gösterimlerini (embeddings) hesaplar.

        Args:
            labels (List[str]): Sınıflandırma etiketlerinin listesi.
        """
        if not self.model:
            print("Model yüklenmediği için etiketler ayarlanamıyor.")
            return

        self.labels = labels
        self.label_embeddings = self.model.encode(self.labels, convert_to_tensor=True)

    def classify(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Verilen metni en benzer etikete göre sınıflandırır.

        Args:
            text (str): Sınıflandırılacak metin.
            threshold (float): Benzerlik skoru için minimum eşik değeri.

        Returns:
            Dict: Tahmin edilen etiket ve benzerlik skorunu içeren bir sözlük.
        """
        if not self.model:
            return {"label": "diğer", "score": 0.0}

        if not self.labels:
            raise ValueError("Sınıflandırma etiketleri belirlenmemiş. Önce set_labels() metodunu çağırın.")

        text_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embedding, self.label_embeddings)[0]

        best_match_index = np.argmax(cosine_scores.cpu().numpy())
        best_score = cosine_scores[best_match_index].item()
        predicted_label = self.labels[best_match_index]

        if best_score >= threshold:
            return {"label": predicted_label, "score": best_score}
        else:
            return {"label": "diğer", "score": best_score}

    def classify_with_multiple_matches(self, text: str, top_k: int = 3) -> List[Dict]:
        """
        Verilen metin için en yüksek skora sahip birden fazla etiketi döndürür.

        Args:
            text (str): Sınıflandırılacak metin.
            top_k (int): Döndürülecek en iyi eşleşme sayısı.

        Returns:
            List[Dict]: Her bir eşleşme için etiket ve skoru içeren sözlük listesi.
        """
        if not self.model:
            return []

        if not self.labels:
            raise ValueError("Sınıflandırma etiketleri belirlenmemiş. Önce set_labels() metodunu çağırın.")

        text_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embedding, self.label_embeddings)[0]

        top_results = np.argpartition(-cosine_scores.cpu().numpy(), top_k)[0:top_k]

        results = []
        for index in top_results:
            results.append({
                "label": self.labels[index],
                "score": cosine_scores[index].item()
            })

        return results