📚 Film Buddy

![Film Buddy Logo](assets/film_buddy_logo.png)

**Knowledge Scout**, RAG (Retrieval-Augmented Generation) mimarisiyle çalışan, Türkçe dil desteğine sahip bir soru-cevap sistemidir. CSV veya TXT dosyalarından veri işler, vektör veritabanında saklar ve kullanıcının sorularına yapay zeka destekli yanıtlar üretir. Sistemin en önemli özelliklerinden biri, daha doğru sonuçlar için **niyet sınıflandırma** yapması ve performans takibi için **dahili bir sayaç** içermesidir.

⚙️ Özellikler

* **Çoklu Veri Formatı Desteği:** `.csv` ve `.txt` dosyalarını okuyabilir.
* **Akıllı Niyet Sınıflandırma:** Kullanıcının amacını (`öneri`, `arama`, `gerçek` gibi) analiz ederek daha odaklı ve isabetli yanıtlar sağlar.
* **LLM Kullanım Sayacı:** Her LLM çağrısını sayarak sistemin performansını ve maliyetini takip etmenize olanak tanır.
* **LangChain Entegrasyonu:** Doküman işleme, parçalara ayırma ve tüm RAG zincirini yönetmek için kullanılır.
* **HuggingFace Embeddings:** Verileri vektörlere dönüştürerek hızlı ve alakalı arama yapılmasını sağlar.
* **ChromaDB:** Vektörleri kalıcı olarak saklayan hafif ve etkili bir veritabanıdır.
* **EnsembleRetriever:** Hem anlamsal (Chroma) hem de anahtar kelime tabanlı (BM25) aramayı birleştirerek sonuçların kalitesini artırır.
* **Esnek LLM Yapısı:** Ücretsiz modellere erişim sağlayan **Gpt4FreeLLM** ve OpenAI API'si ile çalışan **CustomLLM** seçenekleri bulunur.
* **Sohbet Geçmişi Yönetimi:** Diyalog boyunca bağlamı korur ve istenildiğinde geçmişi temizleme komutu sunar.
* **Tam Etkileşimli Kullanım:** Komut satırı arayüzü sayesinde kolay ve hızlı bir deneyim sunar.

📂 Proje Yapısı

```
knowledge-scout/
│
├── data_handler.py      # Veri yükleme ve Doküman oluşturma
├── llm_model.py         # LLM'leri yönetir ve sayaç içerir
├── counter.py           # Performans metriklerini takip eden Counter sınıfı
├── rag_system.py        # RAG hattını ve niyet sınıflandırma mantığını barındırır
├── main.py              # Uygulamanın ana giriş noktası
├── requirements.txt     # Proje bağımlılıkları
└── doc/
    └── n_movies.csv     # Örnek veri dosyası
```

🛠 Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/zeynepcagil/knowledge-scout.git
cd knowledge-scout
```

2. **Gereklilikleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **(İsteğe Bağlı) OpenAI API anahtarınızı ayarlayın:** `CustomLLM` kullanmak isterseniz `main.py` içindeki ilgili satırı güncelleyin.
```python
openai_api_token = "SİZİN_API_ANAHTARINIZ"
```

🚀 Kullanım

1. **Ana dosyayı çalıştırın:**
```bash
python main.py
```

2. **Kendi verinizi kullanmak için:** `main.py` dosyasındaki `csv_path` değişkenini kendi `.csv` veya `.txt` dosyanızın yoluna göre değiştirin.
3. **Komutlar:**
   * `q`: Programdan çıkış yapar.
   * `temizle`: Sohbet geçmişini sıfırlar.
   * Diğer tüm girdiler sisteme soru olarak gönderilir.

📌 Örnek Kullanım

```bash
Sorunuzu giriniz: film öner
Asistan: Harika! Bir film/dizi önerisi arıyorsunuz. Nasıl bir türde istersiniz? Mesela, 'aksiyon filmi' ya da 'romantik komedi' gibi.
--- LLM çağrı sayısı: 1
Sorunuzu giriniz: romantik komedi
Asistan: 2018'den sonra çıkan romantik komediler arasında ne tür bir konu arıyorsunuz?
--- LLM çağrı sayısı: 2
```

---

## Detaylı Kullanım Örneği

```
New g4f version available: 0.6.0.2 (current: 0.6.0.1) | pip install -U g4f

Sorunuzu giriniz: canım sıkkın bana film öner
Debug - Yeniden yazılan sorgu: movies
Asistan: Hangi tür filmlerden hoşlanırsınız?
---
Toplam LLM çağrısı: 1
Toplam token kullanımı: 705 (Giriş: 594, Çıkış: 111)

Sorunuzu giriniz: komedi filmlerinden hoşlanırım
Debug - Yeniden yazılan sorgu: comedy movies
Asistan: Bağlamda belirtilen bilgilere göre, komedi türünde iki film bulunmaktadır:
1. **Jim Gaffigan: Cinco**  
   - **Yıl:** 2017  
   - **Tür:** Komedi  
   - **Süre:** 73 dakika  
   - **Rating:** 7.1  
2. **The Hall: Honoring the Greats of Stand-Up**  
   - **Yıl:** 2022  
   - **Tür:** Komedi  
   - **Süre:** 70 dakika  
   - **Rating:** 6.3  
Bu filmler komedi türüne uygun olarak önerilebilir.
---
Toplam LLM çağrısı: 3
Toplam token kullanımı: 2845 (Giriş: 2496, Çıkış: 349)

Sorunuzu giriniz: the hallın konusu ne
Debug - Yeniden yazılan sorgu: The Hall: Honoring the Greats of Stand-Up plot
Asistan: "The Hall: Honoring the Greats of Stand-Up" filminin konusu, stand-up komedi alanında büyük başarılar elde etmiş isimleri onurlandırmak ve komedi severleri bir araya getirmek üzerine kuruludur. Film, bu alanda öne çıkan komedyenlerin katkılarını kutlar ve izleyicilere bir kutlama atmosferi sunar.
---
Toplam LLM çağrısı: 6
Toplam token kullanımı: 5747 (Giriş: 5188, Çıkış: 559)

Sorunuzu giriniz: puanı kaç
Debug - Yeniden yazılan sorgu: The Hall: Honoring the Greats of Stand-Up rating
Asistan: "The Hall: Honoring the Greats of Stand-Up" filminin ratingi 6.3'tür.
---
Toplam LLM çağrısı: 9
Toplam token kullanımı: 8684 (Giriş: 7990, Çıkış: 694)

Sorunuzu giriniz: q
Uygulamadan çıkılıyor.
Process finished with exit code 0
```

---

🤝 Katkıda Bulunma

Projeye katkıda bulunmak için:
1. Bir `fork` oluşturun.
2. Yeni bir `branch` açın.
3. Değişikliklerinizi `commit`'leyin.
4. Bir `pull request` gönderin.
