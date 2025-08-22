import os
import random
from typing import List, Dict

from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from classifiers.sentence_transformer_classifier import SentenceTransformerClassifier

# ChromaDB telemetrisini devre dışı bırak
os.environ["CHROMA_TELEMETRY_SETTINGS"] = '{"anonymized_telemetry": false}'


class RagSystem:
    """
    Film ve dizi önerileri yapmak için RAG (Retrieval-Augmented Generation)
    sistemini yöneten ana sınıf.
    """

    def __init__(self, documents: List[Document], llm: LLM, db_path: str = "./chroma_db_bge_csv"):
        self.documents = documents
        self.llm = llm
        self.db_path = db_path
        self.qa_chain = None
        self.retriever = None
        self.memory = None
        self.classifier = SentenceTransformerClassifier()
        self.conversation_history = []
        self.chat_history_with_queries = []
        self.intent_examples = self._load_intent_examples()
        self.initialize_pipeline()

    def _load_intent_examples(self) -> Dict[str, List[str]]:
        """Intent sınıflandırması için örnek cümleleri yükler."""
        # 'lookup' intent'ini bağlamsal soruları da içerecek şekilde genişletiyoruz.
        return {
            "recommendation": [
                "hangi filmi izleyebilirim", "aksiyon filmi öner", "komedi dizisi istiyorum",
                "romantik film", "korku filmi öner", "bilim kurgu dizisi",
                "thriller film", "animasyon öner", "netflix dizisi",
                "yeni çıkan filmler", "klasik filmler", "dizi önerisi",
                "film tavsiyesi", "iyi film var mı", "güzel dizi",
                "izlemelik film", "drama filmi", "macera filmi", "aile filmi"
            ],
            "lookup": [
                "bu filmi tanıyor musun", "film hakkında bilgi", "dizinin konusu nedir",
                "film özeti", "oyuncular kimler", "yönetmen kim",
                "film ne zaman çıktı", "kaç sezon var", "film detayları",
                "dizi bilgileri", "cast bilgisi", "filmin imdb puanı", "dizi kaç bölüm",
                "ne hakkında", "konusu ne", "hakkında bilgi",
                "puanı kaç", "imdb puanı ne", "yılı kaç", "kim oynuyor", "oyuncuları kim",
                "ne kadar sürdü", "kaç bölüm", "o neydi"
            ],
            "greeting": [
                "merhaba", "selam", "günaydın", "iyi günler", "nasılsın",
                "naber", "teşekkürler", "sağol", "hoşçakal", "merhabalar",
                "iyiyim", "kötüyüm", "harikayım", "yorgunum"
            ],
            "other": [
                "hava durumu", "matematik problemi", "tarif ver", "para kazanma",
                "sağlık tavsiyeleri", "spor haberleri", "siyaset", "ekonomi",
                "teknoloji haberleri", "oyun öner", "kitap öner", "müzik öner",
                "alışveriş", "seyahat", "iş bulma", "ders çalışma",
                "python kodlama", "resim çiz", "şarkı sözleri"
            ]
        }

    def classify_intent(self, query: str) -> Dict[str, any]:
        """Kullanıcı sorgusunun niyetini (intent) belirler."""
        query_lower = query.lower()
        results = []
        # Her bir intent için sınıflandırma yap ve sonuçları topla
        for intent, examples in self.intent_examples.items():
            self.classifier.set_labels(examples)
            result = self.classifier.classify(query_lower, threshold=0.4)
            results.append((intent, result['score'] if result['label'] != 'diğer' else 0.0))

        # En yüksek skora sahip intent'i seç
        best_intent, best_score = max(results, key=lambda x: x[1])

        # Eğer en iyi skor 0'sa (hiçbir intent eşleşmediyse)
        if best_score == 0.0:
            self.classifier.set_labels(self.intent_examples['greeting'])
            greeting_result_low = self.classifier.classify(query_lower, threshold=0.2)
            if greeting_result_low['label'] != 'diğer':
                best_intent = "greeting"
                best_score = greeting_result_low['score']
            else:
                best_intent = "other"
                best_score = 0.0

        needs_clarification = False
        if best_intent == 'recommendation' and best_score < 0.6:
            genres = ['aksiyon', 'komedi', 'romantik', 'korku', 'drama', 'macera']
            if not any(genre in query_lower for genre in genres):
                needs_clarification = True

        return {
            'intent': best_intent,
            'confidence': best_score,
            'needs_clarification': needs_clarification
        }

    def initialize_pipeline(self):
        """Sistemin RAG bileşenlerini (vektör veritabanı, retriever, chain) başlatır."""
        print("Sistem başlatılıyor...")
        device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
        print(f"Embedding modeli yükleniyor... (device={device})")

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device}
        )

        if os.path.exists(self.db_path):
            print("Veri tabanı bulundu, yükleniyor...")
            db = Chroma(persist_directory=self.db_path, embedding_function=embeddings)
            existing_docs = db._collection.count()
            if existing_docs < len(self.documents):
                print("Yeni veriler bulundu, ekleniyor...")
                new_docs = self.documents[existing_docs:]
                chunks = self._split_documents(new_docs)
                db.add_documents(chunks)
                db.persist()
        else:
            print("Yeni veri tabanı oluşturuluyor...")
            chunks = self._split_documents(self.documents)
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.db_path
            )
            db.persist()

        print("Veri tabanı hazır.")
        chunks = self._split_documents(self.documents)
        basic_retriever = db.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        self.retriever = EnsembleRetriever(
            retrievers=[basic_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )

        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )

        custom_prompt_template = """Sen, kullanıcıya film ve dizi öneren samimi, içten ve yaratıcı bir asistanssın. Kullanıcıya hitap ederken doğal bir sohbet dilini kullan. Cevaplarında kalıplaşmış, robotik ifadelerden ve genelleyici cümlelerden kaçın.

        Önerdiğin filmleri veya dizileri, sanki o eseri gerçekten izlemiş ve beğenmiş bir arkadaşın gibi anlat. Önerinin hemen başında kullanıcının isteğine uygun, kişisel bir giriş yap.

        Örnekler:
        - "Harika bir tercih! Komedi filmlerine bayılıyorum. Sizin için bir tane buldum: [Film Adı]. Animasyon tarzı, yetişkinlere yönelik bir film arıyorsanız, bu tam size göre olabilir."
        - "Aksiyon filmlerinde adrenalin çok önemlidir, değil mi? Tam da aradığınız gibi bir film buldum: [Film Adı]. Baştan sona temposu hiç düşmeyen, aksiyon dolu bir macera."
        - "Ah, bu filmin konusu gerçekten çok ilginç. [Film Adı] hakkında size biraz bilgi vereyim..."

        Sadece aşağıdaki bağlamda bulunan bilgileri kullan. Bağlamda bir film varsa, o filmin sorudaki kriterlere (tür, yıl, vb.) uyup uymadığını kontrol et ve uygunsa öner.
        Eğer gelen bağlamda film/dizi önerisi yoksa veya aradığınız kriterlere uygun bir film bulunamıyorsa, "Aradığınız kriterlere uygun bir film bulamadım. Başka bir tür veya farklı bir arama yapmayı dener misiniz?" de.

        Sohbet Geçmişi:
        {chat_history}

        Bağlam:
        {context}

        Soru:
        {question}

        Cevap:"""
        self.qa_chain_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["chat_history", "context", "question"]
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_chain_prompt}
        )
        print("Sistem kullanıma hazır.")

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Dokümanları daha küçük parçalara (chunk) böler."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(docs)

    def clear_chat_history(self):
        """Sohbet geçmişini temizler."""
        if self.memory:
            self.memory.clear()
            print("Sohbet geçmişi temizlendi.")

    def _extract_constraints_from_query(self, query: str) -> str:
        """Sorgudan türleri ve diğer kısıtlamaları çıkarıp sorguyu zenginleştirir."""
        genre_mapping = {
            'aksiyon': 'action', 'komedi': 'comedy', 'romantik': 'romance',
            'korku': 'horror', 'bilim kurgu': 'sci-fi', 'gerilim': 'thriller',
            'drama': 'drama', 'animasyon': 'animation', 'belgesel': 'documentary',
            'macera': 'adventure', 'suç': 'crime', 'savaş': 'war'
        }
        enhanced_query = query.lower()
        detected_genres = []
        for tr_genre, en_genre in genre_mapping.items():
            if tr_genre in enhanced_query:
                detected_genres.append(en_genre)
        if 'dizi' in enhanced_query and 'film' not in enhanced_query:
            enhanced_query += " series"
        elif 'film' in enhanced_query and 'dizi' not in enhanced_query:
            enhanced_query += " movie"
        if detected_genres:
            enhanced_query += f" {' '.join(detected_genres)}"
        return enhanced_query

    def _handle_other_intent(self) -> Dict:
        """Film/dizi dışı konulara verilen yanıtları yönetir."""
        other_responses = [
            "Hmm, bu konuda pek bilgim yok ama film önerebilirim! Ne dersin? 🎬",
            "O konu benim uzmanlık alanım değil ama filmlerden çok iyi anlarım! 😊",
            "Bu konuyu bilmiyorum ama sana güzel filmler bulabilirim! İster misin?",
            "Maalesef o konuda yardımcı olamam ama film konusunda harikulade tavsiyelerim var! 🍿"
        ]
        return {"result": random.choice(other_responses)}

    def _handle_greeting(self, query: str) -> Dict:
        """Selamlama ve veda ifadelerini daha doğal şekilde ele alır."""
        farewell_keywords = ['görüşürüz', 'hoşça kal', 'bye', 'bb']
        if any(word in query.lower() for word in farewell_keywords):
            farewell_responses = [
                "Hoşça kal! Film izlerken keyifli vakit geçir! 🎬",
                "Görüşmek üzere! İyi filmler! 🍿",
                "Kendine iyi bak! Umarım önerdiğim filmler hoşuna gider 😊"
            ]
            return {"result": random.choice(farewell_responses)}

        greeting_result = self.classifier.classify(query.lower(), threshold=0.4)
        response_sets = {
            'merhaba': ["Merhaba! Ne tür filmler seversin?", "Selam! Bugün hangi ruh halinde film izlemek istiyorsun?"],
            'teşekkürler': ["Rica ederim! İyi seyirler dilerim! 🍿", "Ne demek! Başka film lazım olursa söyle 😊"],
            'nasılsın': ["İyiyim, teşekkürler! Sen nasılsın? Film modunda mısın?"],
            'iyiyim': ["Süper! O zaman güzel filmler bulalım sana!", "Harika! Ruh haline uygun film önereyim mi?"]
        }
        label = greeting_result['label']
        if label != 'diğer' and label in response_sets:
            return {"result": random.choice(response_sets[label])}
        return {"result": "Merhaba! Film önerisi mi arıyorsun?"}

    def _check_for_duplicate_query(self, new_query: str, threshold: float = 0.95) -> str:
        """
        Yeni sorgunun, geçmişteki sorgulara çok benzeyip benzemediğini kontrol eder.
        """
        if not self.chat_history_with_queries:
            return None
        past_queries = [item['query'] for item in self.chat_history_with_queries]
        self.classifier.set_labels(past_queries)
        result = self.classifier.classify(new_query.lower(), threshold=threshold)
        if result['label'] != 'diğer' and result['score'] >= threshold:
            matched_index = past_queries.index(result['label'])
            return self.chat_history_with_queries[matched_index]['response']
        return None

    def ask(self, query: str) -> Dict:
        """Kullanıcının sorgusunu işler ve yanıt verir."""
        if not self.qa_chain:
            return {"result": "Sistem şu an hazır değil, lütfen bekleyin."}

        duplicate_response = self._check_for_duplicate_query(query)
        if duplicate_response:
            print("Debug: Benzer sorgu tespit edildi, eski cevap döndürülüyor.")
            return {"result": duplicate_response}

        intent_result = self.classify_intent(query)
        user_intent = intent_result['intent']
        confidence = intent_result['confidence']
        print(f"Debug: Intent={user_intent}, Confidence={confidence:.3f}")

        # Eğer niyet bir öneri ise, sorgunun yeterli detay içerip içermediğini kontrol et.
        if user_intent == 'recommendation':
            # Sorguyu zenginleştirmek için kullanılan anahtar kelimeleri kontrol et.
            enhanced_query = self._extract_constraints_from_query(query)

            # Eğer sorgu sadece "film öner" gibi genel bir ifade ise (ve zenginleştirme sonucu değişmemişse)
            # o zaman kullanıcıdan ek bilgi iste.
            if enhanced_query == query.lower():
                clarification_responses = [
                    "Hangi tür filmlerden hoşlanırsın? Mesela, aksiyon, komedi, romantik ya da bilim kurgu? ",
                    "Dizi mi film mi? Ya da ne tür bir ruh halinde olduğunu söyle, sana ona göre bir şeyler bulalım. ",
                    "Canın ne çekiyor? Macera, gerilim ya da belki biraz drama? ",
                    "Nasıl bir film izlemek istersin? Komik mi, heyecan verici mi yoksa düşündürücü mü?"
                ]
                return {"result": random.choice(clarification_responses)}

            # Sorgu yeterince spesifikse, RAG pipeline'ını çalıştır.
            try:
                response = self.qa_chain.invoke({"question": enhanced_query})
                result = response.get("answer", "Bir sorun oluştu.")

                if "Aradığınız kriterlere uygun" in result or "bulamadım" in result.lower():
                    not_found_responses = [
                        "Hmm, tam istediğin gibi film bulamadım. Başka bir tür deneyelim mi? 🤔",
                        "Bu kriterlere uygun film çıkmadı. Biraz farklı bir arama yapalım mı?",
                    ]
                    result = random.choice(not_found_responses)

                self.chat_history_with_queries.append({'query': query.lower(), 'response': result})
                return {"result": result}
            except Exception as e:
                print(f"RAG pipeline hatası: {e}")
                return {"result": "Üzgünüm, bir hata oluştu. Daha sonra tekrar deneyin."}

        # Bilgi arama (lookup) intent'i için RAG pipeline'ını çalıştır.
        elif user_intent == 'lookup':
            try:
                response = self.qa_chain.invoke({"question": query})
                result = response.get("answer", "Bir sorun oluştu.")

                if "Aradığınız kriterlere uygun" in result or "bulamadım" in result.lower():
                    not_found_responses = [
                        "Üzgünüm, bu konuda yeterli bilgi bulamadım. Başka bir şey sormak ister misin?",
                        "Bu bilgiye sahip değilim, ama sana film veya dizi önerebilirim!",
                    ]
                    result = random.choice(not_found_responses)

                self.chat_history_with_queries.append({'query': query.lower(), 'response': result})
                return {"result": result}
            except Exception as e:
                print(f"RAG pipeline hatası: {e}")
                return {"result": "Üzgünüm, bir hata oluştu. Daha sonra tekrar deneyin."}

        # Diğer intent'ler için önceden tanımlı yanıtlar kullanılır.
        elif user_intent == 'other':
            return self._handle_other_intent()
        elif user_intent == 'greeting':
            return self._handle_greeting(query)

        # Olası bir hata durumunda genel bir yanıt.
        return {
            "result": "Üzgünüm, isteğinizi tam olarak anlayamadım. Film veya dizi önerisi için ne tür bir şey izlemek istediğinizi söyleyebilir misiniz?"}
