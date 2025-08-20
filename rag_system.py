import os
import json
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
from classifiers.sentence_transformer_classifier import SentenceTransformerClassifier  # Yeni import

os.environ["CHROMA_TELEMETRY_SETTINGS"] = '{"anonymized_telemetry": false}'


class RagSystem:
    def __init__(self, documents: List[Document], llm: LLM, db_path="./chroma_db_bge_csv"):
        self.documents = documents
        self.llm = llm
        self.db_path = db_path
        self.qa_chain = None
        self.retriever = None
        self.memory = None
        self.classifier = SentenceTransformerClassifier()  # Yeni eklenen sınıf
        self.conversation_history = []
        # Dinamik soru sorma için temel kısıtlamalar ve sorular
        self.default_clarifications = [
            "Hangi türde film/dizi arıyorsunuz?",
            "Daha çok hangi türleri seversiniz?",
            "Dizi mi yoksa film mi arıyorsunuz?"
        ]

        # Yeni ve daha esnek bir prompt
        self.intent_prompt_template = PromptTemplate(
            template="""Sen, kullanıcının ruh haline ve detaylı tercihlerine göre kişiselleştirilmiş film ve dizi önerileri sunan bir asistansın.
            Sadece geçerli JSON formatında yanıt ver. Başka bir metin ekleme.

            Kullanıcının niyetini şu kategorilerden biri olarak sınıflandır:
            recommendation | lookup | fact | clarification | other.

            Sohbet geçmişindeki bilgileri (tür, film/dizi ayrımı, ruh hali, yıl, ülke vb.) BİRLEŞTİREREK arama sorgusunu ve kısıtlamaları güncelle.
            Eğer kullanıcı yeni bir kısıtlama (örn. yıl, tür) eklerse, bunu önceki sorguyla harmanla.
            Örneğin, kullanıcı 'dizi olsun' dedikten sonra 'komedi' derse, sorguyu 'komedi dizileri' olarak yeniden yaz.
            Aksiyon filmi aradıktan sonra '2020den sonra çıkanı söyle' derse, sorguyu 'aksiyon filmleri 2020 sonrası' olarak yeniden yaz.

            Yoğun arama (dense retrieval) için EKSTRA kelime olmadan KISA ve odaklı bir İngilizce arama sorgusu üret.
            Kullanıcı metninden yapısal kısıtlamaları (yıllar, türler, ülkeler, süre, puan) İngilizceye çevir.

            Eğer bir film/dizi önerisi için temel bilgi (tür, film/dizi ayrımı vb.) eksikse, 'needs_clarification' değerini 'true' yap.
            Bu durumda, 'clarification_notes' alanına kullanıcıya sorulacak Türkçe, net ve yönlendirici sorular ekle.

            Verilen şemaya uygun SADECE geçerli JSON döndür.
            JSON formatı:
            {{
              "intent": "...",
              "rewritten_query": "...",
              "constraints": {{
                "genres": ["..."],
                "year": "...",
                "country": "...",
                "duration": "...",
                "rating": "..."
              }},
              "needs_clarification": false,
              "clarification_notes": ["..."]
            }}

            Sohbet Geçmişi:
            {chat_history}
            Kullanıcı: {user_input}
            JSON:
            """,
            input_variables=["user_input", "chat_history"]
        )

    def initialize_pipeline(self):
        print("Sistem başlatılıyor...")
        device = "cuda" if os.system("nvidia-smi > nul 2>&1") == 0 else "cpu"
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

        custom_prompt_template = """Aşağıdaki sohbet geçmişi ve bağlamı kullanarak soruya cevap ver.
        Cevabını oluştururken kendi düşünce sürecini yazma.
        Güncel bilgilere ulaşmaya çalışma.
        YANLIZCA VE SADECE AŞAĞIDAKİ BAĞLAMDA BULUNAN BİLGİLERİ KULLAN. Bağlamda bir film varsa, o filmin sorudaki kriterlere (tür, yıl, vb.) uyup uymadığını kontrol et ve uygunsa öner.
        Eğer gelen bağlamda film/dizi önerisi yoksa veya aradığınız kriterlere uygun bir film bulunamıyorsa, "Üzgünüm, aradığınız kriterlere uygun bir film bulamadım. Başka bir tür veya farklı bir arama yapmayı dener misiniz?" de.
        Soru, bağlamda belirtilen kriterlere uygun bir film olup olmadığını kontrol etmek için tasarlanmıştır. Bu kontrolü yap ve sonucu bildir.

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

    def _split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(docs)

    def ask(self, query: str) -> Dict:
        if not self.qa_chain:
            return {"result": "Sistem şu an hazır değil, lütfen bekleyin."}

        # Tekrarlayan soru kontrolü
        chat_history_list = self.memory.chat_memory.messages
        past_queries = [msg.content for msg in chat_history_list if msg.type == "human"]

        if past_queries:
            self.classifier.set_labels(past_queries)
            # Eşik değeri biraz daha düşürülebilir, 0.7 gibi bir değer daha esnek olabilir.
            classification_result = self.classifier.classify(query, threshold=0.7)

            if classification_result['label'] != 'diğer' and classification_result['score'] > 0.7:
                matched_query = classification_result['label']

                matched_response = ""
                for i in range(len(chat_history_list)):
                    if chat_history_list[i].type == "human" and chat_history_list[i].content == matched_query:
                        if i + 1 < len(chat_history_list) and chat_history_list[i + 1].type == "ai":
                            matched_response = chat_history_list[i + 1].content
                            break

                if matched_response:
                    return {"result": f"Bu soruyu zaten sormuştunuz. Eski cevabım: {matched_response}"}

        # Eğer benzer soru bulunmazsa, normal akışı takip et
        try:
            chat_history_str = self._get_chat_history_as_string()
            intent_response_str = self.llm.invoke(
                self.intent_prompt_template.format(user_input=query, chat_history=chat_history_str))

            json_start = intent_response_str.find('{')
            json_end = intent_response_str.rfind('}')

            if json_start == -1 or json_end == -1:
                print(f"Uyarı: LLM'den geçerli JSON alınamadı. Ham çıktı: {intent_response_str}")
                intent_data = {"intent": "other", "rewritten_query": query, "needs_clarification": False,
                               "clarification_notes": []}
            else:
                json_str_cleaned = intent_response_str[json_start:json_end + 1]
                intent_data = json.loads(json_str_cleaned)

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Niyet sınıflandırma hatası: {e}")
            return {
                "result": "Üzgünüm, şu anda isteğinizi anlamakta zorlanıyorum. Lütfen daha açık bir ifade kullanır mısınız?"}

        user_intent = intent_data.get("intent")
        needs_clarification = intent_data.get("needs_clarification")
        clarification_notes = intent_data.get("clarification_notes", [])
        rewritten_query = intent_data.get("rewritten_query", query)

        print(f"Debug - Yeniden yazılan sorgu: {rewritten_query}")

        if needs_clarification and clarification_notes:
            return {"result": clarification_notes[0]}

        if user_intent == "recommendation":
            constraints = intent_data.get("constraints", {})
            has_constraints = any(constraints.get(key) for key in ["genres", "year", "country", "duration", "rating"])
            if not has_constraints:
                return {
                    "result": "Harika! Bir film/dizi önerisi arıyorsunuz. Nasıl bir türde istersiniz? Mesela, 'aksiyon filmi' ya da 'romantik komedi' gibi."}

        if not rewritten_query or not rewritten_query.strip():
            print("Uyarı: Yeniden yazılan sorgu boş kaldı.")
            return {"result": "Üzgünüm, isteğinizi tam olarak anlayamadım. Lütfen daha detaylı bilgi verir misiniz?"}

        try:
            response = self.qa_chain.invoke({"question": rewritten_query})

            if isinstance(response, dict) and "answer" in response:
                result = response["answer"]
            else:
                print(f"Uyarı: QA Chain'den geçersiz yanıt formatı alındı: {response}")
                return {"result": "İsteğiniz işlenirken bir sorun oluştu. Lütfen tekrar deneyin."}
        except Exception as e:
            print(f"QA Chain çağrılırken hata oluştu: {e}")
            return {"result": "AI servisi şu anda yanıt vermiyor. Lütfen tekrar deneyin."}

        if "Üzgünüm, aradığınız bilgiye elimdeki verilerle ulaşamıyorum" in result:
            return {
                "result": "Aradığınız kriterlere uygun bir film bulamadım. Başka bir tür veya farklı bir arama yapmayı dener misiniz?"}

        # Sonucu döndürmeden önce, sohbet geçmişini yeniden yazılan sorgu ve yanıtla güncelleyin.
        self.memory.chat_memory.add_user_message(rewritten_query)
        self.memory.chat_memory.add_ai_message(result)

        return {"result": result}

    def _get_chat_history_as_string(self):
        """Sohbet geçmişini LLM'e verilecek string formatına dönüştürür."""
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        return " ".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    def clear_chat_history(self):
        if self.memory:
            self.memory.clear()
            print("Sohbet geçmişi temizlendi.")