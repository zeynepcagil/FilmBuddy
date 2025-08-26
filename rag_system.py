import os
import random
from ctypes import util
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
        self.intent_examples = self._load_intent_examples()
        self.conversation_state = {
            'user_greeted': False,
            'mood_shared': False,
            'preference_asked': False,
            'last_intent': None,
            'consecutive_greetings': 0,
            'asked_questions': [],
            'last_responses': {},
            'current_context': None
        }
        self.initialize_pipeline()

    def _load_intent_examples(self) -> Dict[str, List[str]]:
        """Intent sınıflandırması için örnek cümleleri yükler."""
        return {
            "recommendation": [
                "hangi filmi izleyebilirim", "aksiyon filmi öner", "komedi dizisi istiyorum",
                "romantik film", "korku filmi öner", "bilim kurgu dizisi",
                "thriller film", "animasyon öner", "netflix dizisi",
                "yeni çıkan filmler", "klasik filmler", "dizi önerisi",
                "film tavsiyesi", "iyi film var mı", "güzel dizi",
                "izlemelik film", "drama filmi", "macera filmi", "aile filmi",
                "komedi filmi öner", "aksiyon istiyorum", "romantik dizi"
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
            "mood_expression": [
                "canım sıkılıyor", "canım sıkkın", "üzgünüm", "mutsuzum",
                "keyifli hissediyorum", "enerjiliyim", "yorgunum", "stresli",
                "rahatlamak istiyorum", "eğlenmek istiyorum", "ağlamak istiyorum",
                "gülmek istiyorum", "heyecanlı", "sakin", "melankolik"
            ],
            "greeting": [
                "merhaba", "selam", "günaydın", "iyi günler", "nasılsın",
                "naber", "teşekkürler", "sağol", "hoşçakal", "merhabalar",
                "iyiyim", "kötüyüm", "harikayım", "yorgunum", "hi", "hello"
            ],
            "affirmative": [
                "evet", "tabii", "olur", "tamam", "istiyorum", "isterim",
                "kesinlikle", "aynen", "doğru", "kabul", "yes", "okay"
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

        # Önce film/dizi dışı istekleri kontrol et
        non_film_keywords = [
            'yemek tarifi', 'tarif', 'yemek', 'nasıl yapılır', 'malzeme',
            'hava durumu', 'matematik', 'ders', 'ödev', 'kod', 'program',
            'para kazanma', 'iş', 'sağlık', 'doktor', 'hastalık', 'spor',
            'siyaset', 'ekonomi', 'teknoloji', 'oyun', 'kitap', 'müzik',
            'alışveriş', 'seyahat', 'python', 'resim çiz', 'şarkı'
        ]

        if any(keyword in query_lower for keyword in non_film_keywords):
            return {'intent': 'other', 'confidence': 0.9, 'needs_clarification': False}

        results = []

        # Sonra spesifik tür anahtar kelimelerini kontrol et
        specific_genre_keywords = ['komedi', 'aksiyon', 'romantik', 'korku', 'drama', 'bilim kurgu']
        if any(genre in query_lower for genre in specific_genre_keywords):
            return {'intent': 'recommendation', 'confidence': 0.9, 'needs_clarification': False}

        for intent, examples in self.intent_examples.items():
            self.classifier.set_labels(examples)
            result = self.classifier.classify(query_lower, threshold=0.4)
            if result['label'] != 'diğer':
                results.append((intent, result['score']))

        if not results:
            return {'intent': 'other', 'confidence': 0.0, 'needs_clarification': False}

        best_intent, best_score = max(results, key=lambda x: x[1])

        # Konuşma bağlamına göre intent'i ayarla
        if self.conversation_state['last_intent'] == 'greeting' and best_intent == 'greeting':
            if any(word in query_lower for word in ['evet', 'tabii', 'istiyorum']):
                best_intent = 'affirmative'

        needs_clarification = False
        if best_intent == 'recommendation' and best_score < 0.6:
            genres = ['aksiyon', 'komedi', 'romantik', 'korku', 'drama', 'macera', 'bilim kurgu']
            if not any(genre in query_lower for genre in genres):
                needs_clarification = True

        lookup_keywords = ['adı ne', 'kim oynuyor', 'konusu ne', 'puanı kaç', 'imdb puanı']
        if any(keyword in query_lower for keyword in lookup_keywords):
            best_intent = 'lookup'

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
            k=5,
            memory_key="chat_history",
            return_messages=True
        )

        custom_prompt_template = """Sen, kullanıcıya film ve dizi öneren samimi, içten ve yaratıcı bir asistanssın. Kullanıcıya hitap ederken doğal bir sohbet dilini kullan. Cevaplarında kalıplaşmış, robotik ifadelerden ve genelleyici cümlelerden kaçın.
        Önce kullanıcının ne istediğini anlamaya çalış, direkt öneri verme.
        Context dışına çıkma, yalnızca verilen bağlamı kullan.

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
        self.conversation_history = []
        self.conversation_state = {
            'user_greeted': False,
            'mood_shared': False,
            'preference_asked': False,
            'last_intent': None,
            'consecutive_greetings': 0,
            'asked_questions': [],
            'last_responses': {},
            'current_context': None
        }

    def _check_repeated_question(self, query: str) -> Dict:
        """Aynı sorunun tekrar sorulup sorulmadığını kontrol eder."""
        query_normalized = query.lower().strip()

        normalized_variants = {
            'hakkında': ['ne hakkında', 'konusu ne', 'hakkında bilgi', 'konusu nedir'],
            'puan': ['puanı kaç', 'imdb puanı', 'kaç puan', 'puanı ne'],
            'oyuncu': ['kim oynuyor', 'oyuncular kimler', 'oyuncuları kim'],
            'yönetmen': ['yönetmen kim', 'yöneten kim', 'yönetmeni kim']
        }

        if self.conversation_state['current_context']:
            context_film = self.conversation_state['current_context'].lower()

            for question_type, variants in normalized_variants.items():
                for variant in variants:
                    if variant in query_normalized:
                        question_key = f"{context_film}_{question_type}"

                        if question_key in self.conversation_state['asked_questions']:
                            previous_response = self.conversation_state['last_responses'].get(question_key, "")
                            repeat_responses = [
                                f"Bu soruyu zaten sormuştunuz! {context_film.title()} hakkında şunu söylemiştim: {previous_response}",
                                f"Daha önce de sormuştunuz, {context_film.title()} için yanıtımı tekrar edeyim: {previous_response}",
                                f"Bu konuyu geçen konuştuk, {context_film.title()} hakkında: {previous_response}"
                            ]
                            return {"result": random.choice(repeat_responses)}
                        else:
                            self.conversation_state['asked_questions'].append(question_key)
                            return None

        return None

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

    def _extract_film_context(self, query: str):
        """Sorgudan film/dizi adı gibi bağlam bilgilerini çıkarır ve state'i günceller."""
        # Burada film/dizi adını tespit etmek için daha gelişmiş bir NLP modeli veya
        # önceden tanımlanmış bir liste kullanılabilir. Şimdilik basit bir regex
        # veya anahtar kelime tabanlı yaklaşım kullanabiliriz.

        # Bu sadece bir örnek. Gerçek bir senaryoda daha gelişmiş bir yöntem gerekir.
        film_names = ["inception", "matrix", "fight club", "avengers", "game of thrones"]  # Örnek film listesi
        query_lower = query.lower()

        for name in film_names:
            if name in query_lower:
                self.conversation_state['current_context'] = name.title()
                return

        # Film adı bulunamazsa bağlamı temizle
        self.conversation_state['current_context'] = None

    def _handle_other_intent(self, query: str) -> Dict:
        """Film/dizi dışı konulara verilen yanıtları yönetir."""
        general_responses = [
            "Hmm, bu konuda pek bilgim yok ama film önerebilirim! Ne dersin? 🎬",
            "O konu benim uzmanlık alanım değil ama filmlerden çok iyi anlarım! 😊",
            "Bu konuyu bilmiyorum ama sana güzel filmler bulabilirim! İster misin?",
            "Maalesef o konuda yardımcı olamam ama film konusunda harikulade tavsiyelerim var! 🍿"
        ]

        feeling_responses = {
            "kötü": ["Üzgünüm, kötü hissetmene üzüldüm. Belki seni neşelendirecek bir komedi filmi bulabiliriz?",
                     "Kötü hissetme! Seni güldürecek bir film önereyim mi?"],
            "yorgun": [
                "Anlıyorum, yorgunsun. O zaman seni rahatlatacak, fazla düşünmene gerek kalmayacak bir film bulalım mı?"]
        }

        query_lower = query.lower()

        for keyword, responses in feeling_responses.items():
            if keyword in query_lower:
                return {"result": random.choice(responses)}

        return {"result": random.choice(general_responses)}

    def _handle_mood_expression(self, query: str) -> Dict:
        """Kullanıcının ruh hali ifadelerini ele alır ve uygun önerilerde bulunur."""
        query_lower = query.lower()

        # Önce film/dizi dışı istek var mı kontrol et
        non_film_keywords = [
            'yemek tarifi', 'tarif', 'yemek', 'nasıl yapılır', 'malzeme',
            'hava durumu', 'matematik', 'ders', 'ödev', 'kod', 'program',
            'para kazanma', 'iş', 'sağlık', 'doktor', 'hastalık'
        ]

        if any(keyword in query_lower for keyword in non_film_keywords):
            return self._handle_other_intent(query)

        mood_responses = {
            "sıkkın": [
                "Ah, canın sıkılıyor. Seni neşelendirecek bir komedi filmi önereyim mi? Ya da belki duygusal bir drama tercih edersin?",
                "Canın sıkıldığında iyi bir film çok iyi gelir! Komedi mi yoksa daha duygusal bir şey mi istersin?"
            ],
            "üzgün": [
                "Üzgün hissettiğinde bazen güzel bir drama ya da neşelendirici bir komedi film çok iyi gelir. Hangisini tercih edersin?",
                "Anlıyorum. Bu durumda seni rahatlatacak bir film bulalım. Neşelendirici bir şey mi yoksa duygularını boşaltabileceğin bir drama mı?"
            ],
            "yorgun": [
                "Yorgun olduğunda izlemesi kolay, akışı rahat filmler en iyisi. Hafif bir komedi ya da romantik film nasıl olur?",
                "Yorgunken beyin yorucu filmler izlemek zor. Rahatlatıcı bir şeyler önerebilirim, ne dersin?"
            ],
            "enerjili": [
                "O zaman tempolu bir şeyler lazım! Aksiyon filmi mi yoksa hızlı komedi mi tercih edersin?",
                "Enerjin yüksekse harika aksiyon filmleri önerebilirim. İlgin var mı?"
            ]
        }

        for mood, responses in mood_responses.items():
            if mood in query_lower:
                self.conversation_state['mood_shared'] = True
                return {"result": random.choice(responses)}

        general_mood_responses = [
            "Ruh halini anlıyorum. Hangi tür filmler seni daha iyi hissettirir? Komedi, drama, aksiyon?",
            "Bu durumda doğru film seçimi çok önemli. Ne tür filmler seversin genelde?"
        ]

        self.conversation_state['mood_shared'] = True
        return {"result": random.choice(general_mood_responses)}

    def _check_similar_question(self, query: str, threshold: float = 0.85) -> Dict:
        """Sentence Transformer ile çok benzer soruları tespit eder."""
        if not self.conversation_state.get('asked_questions'):
            return None

        query = query.lower().strip()
        query_emb = self.classifier.model.encode(query, convert_to_tensor=True)

        for prev_q in self.conversation_state['asked_questions']:
            prev_emb = self.classifier.model.encode(prev_q, convert_to_tensor=True)
            sim = util.cos_sim(query_emb, prev_emb).item()
            if sim >= threshold:
                previous_response = self.conversation_state['last_responses'].get(prev_q, "")
                return {
                    "result": f"Bunu zaten sormuştunuz! Önceki yanıtım: {previous_response}"
                }

        # Yeni soruyu kaydet
        self.conversation_state['asked_questions'].append(query)
        return None

    def _handle_greeting(self, query: str) -> Dict:
        """Selamlama ve veda ifadelerini daha akıllı şekilde ele alır."""
        farewell_keywords = ['görüşürüz', 'hoşça kal', 'bye', 'bb', 'bye bye']
        if any(word in query.lower() for word in farewell_keywords):
            farewell_responses = [
                "Hoşça kal! Film izlerken keyifli vakit geçir! 🎬",
                "Görüşmek üzere! İyi filmler! 🍿",
                "Kendine iyi bak! Umarım önerdiğim filmler hoşuna gider 😊"
            ]
            return {"result": random.choice(farewell_responses)}

        query_lower = query.lower()

        if not self.conversation_state['user_greeted']:
            self.conversation_state['user_greeted'] = True
            self.conversation_state['consecutive_greetings'] = 1

            first_greetings = [
                "Merhaba! Film dünyasına hoş geldin! Ne tür filmler seversin?",
                "Selam! Bugün hangi ruh halinde film izlemek istiyorsun?",
                "Merhaba! Size nasıl film önerileri yapabilirim?"
            ]
            return {"result": random.choice(first_greetings)}

        elif self.conversation_state['consecutive_greetings'] > 0:
            self.conversation_state['consecutive_greetings'] += 1

            if self.conversation_state['consecutive_greetings'] >= 3:
                direct_responses = [
                    "Hangi film türünü önereyim? Aksiyon, komedi, drama, romantik... 🎬",
                    "Bugün nasıl bir film izlemek istiyorsun? Neşeli mi, heyecanlı mı, duygusal mı?",
                    "Film önerisi zamanı! Hangi türleri seversin?"
                ]
                self.conversation_state['consecutive_greetings'] = 0
                return {"result": random.choice(direct_responses)}
            else:
                progressive_responses = [
                    "İyi, teşekkürler! Peki hangi film türlerinden hoşlanıyorsun?",
                    "Bende iyiyim! Film izleme konusunda sana nasıl yardım edebilirim?"
                ]
                return {"result": random.choice(progressive_responses)}

        else:
            general_responses = [
                "Selam! Ne tür bir film arıyorsun bugün?",
                "Merhaba! Film önerisi için buradayım!"
            ]
            return {"result": random.choice(general_responses)}

    def _handle_affirmative(self) -> Dict:
        """Kullanıcının 'evet', 'tamam' gibi onaylayıcı yanıtlarını ele alır."""
        if not self.conversation_state['mood_shared'] and not self.conversation_state['preference_asked']:
            preference_questions = [
                "Harika! Hangi film türlerini seversin? Komedi, aksiyon, drama, romantik...?",
                "Süper! Bugün hangi tarzda bir film izlemek istiyorsun?",
                "Mükemmel! Hangi türden filmler seni mutlu eder?"
            ]
            self.conversation_state['preference_asked'] = True
            return {"result": random.choice(preference_questions)}
        else:
            general_responses = [
                "Tamam! Peki hangi türden bir film istersin?",
                "Anlıyorum! Ne tür filmler seversin?"
            ]
            return {"result": random.choice(general_responses)}

    def ask(self, query: str) -> Dict:
        if not self.qa_chain:
            return {"result": "Sistem şu an hazır değil, lütfen bekleyin."}

        self._extract_film_context(query)

        # 1. Önce tekrar soruyu kontrol et
        similar_check = self._check_similar_question(query)
        if similar_check:
            return similar_check

        # 2. Intent sınıflandır
        intent_data = self.classify_intent(query)
        intent = intent_data['intent']
        needs_clarification = intent_data['needs_clarification']
        self.conversation_state['last_intent'] = intent

        # 3. Intentlere göre yanıt
        try:
            if intent == "greeting":
                return self._handle_greeting(query)

            elif intent == "affirmative":
                return self._handle_affirmative()

            elif intent == "mood_expression":
                return self._handle_mood_expression(query)

            elif intent == "other":
                return self._handle_other_intent(query)

            elif intent == "recommendation" and needs_clarification:
                # Kullanıcı ne tür film istediğini belirtmemiş
                return {
                    "result": "Harika! 🎬 Hangi tür filmler ilgini çeker? Komedi, aksiyon, drama, romantik...? Veya bugün hangi ruh halindesin?"
                }

            elif intent == "recommendation" or intent == "lookup":
                self.conversation_state['consecutive_greetings'] = 0

                enhanced_query = self._extract_constraints_from_query(query)
                response = self.qa_chain.invoke({"question": enhanced_query})
                result = response.get("answer", "Bir sorun oluştu.")

                # Film bağlamına göre önceki cevabı kaydet
                if self.conversation_state['current_context']:
                    context_film = self.conversation_state['current_context'].lower()
                    query_lower = query.lower()
                    question_type = None
                    if 'hakkında' in query_lower or 'konusu' in query_lower:
                        question_type = 'hakkında'
                    elif 'puan' in query_lower:
                        question_type = 'puan'
                    elif 'oyuncu' in query_lower:
                        question_type = 'oyuncu'
                    elif 'yönetmen' in query_lower:
                        question_type = 'yönetmen'
                    if question_type:
                        question_key = f"{context_film}_{question_type}"
                        self.conversation_state['last_responses'][question_key] = result

                # Eğer sonuç yoksa yönlendirme mesajı
                if "Aradığınız kriterlere uygun" in result or "bulamadım" in result.lower():
                    not_found_responses = [
                        "Hmm, tam istediğin gibi bir film bulamadım. Başka bir tür deneyelim mi? 🤔",
                        "Bu kriterlere uygun bir sonuç çıkmadı. Belki aramanı biraz daha detaylandırmak istersin?",
                    ]
                    result = random.choice(not_found_responses)

                return {"result": result}

            else:
                return {
                    "result": "Üzgünüm, isteğinizi anlayamadım. Film veya diziyle ilgili bir sorunuz varsa yardımcı olabilirim."
                }

        except Exception as e:
            print(f"Sistem hatası: {e}")
            return {"result": "Üzgünüm, isteğinizi işlerken bir hata oluştu. Lütfen daha sonra tekrar deneyin."}
