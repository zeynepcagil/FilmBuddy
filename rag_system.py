import os
from typing import List, Dict
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

os.environ["CHROMA_TELEMETRY_SETTINGS"] = '{"anonymized_telemetry": false}'


class RagSystem:
    """
    Optimize edilmiş RAG sistemi:
    - İlk yüklemede Chroma ve embedding hesaplar
    - Sonraki yüklemelerde sadece yeni verileri ekler
    - GPU desteği
    """

    def __init__(self, documents: List[Document], llm: LLM, db_path="./chroma_db_bge_csv", use_multiquery=True):
        self.documents = documents
        self.llm = llm
        self.db_path = db_path
        self.qa_chain = None
        self.retriever = None
        self.memory = None
        self.use_multiquery = use_multiquery

    def initialize_pipeline(self):
        """
        RAG pipeline'ını oluşturur ve başlatır.
        """
        # 1️⃣ Embedding modelini yükle (GPU varsa otomatik kullan)
        device = "cuda" if os.system("nvidia-smi > nul 2>&1") == 0 else "cpu"
        print(f"Embedding modeli yükleniyor... (device={device})")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device}
        )

        # 2️⃣ Chroma veritabanı kontrolü
        if os.path.exists(self.db_path):
            print("Chroma veritabanı bulundu, yükleniyor...")
            db = Chroma(persist_directory=self.db_path, embedding_function=embeddings)

            # Yeni veri var mı kontrol et
            existing_docs = db._collection.count()
            if existing_docs < len(self.documents):
                print("Yeni veriler bulundu, ekleniyor...")
                new_docs = self.documents[existing_docs:]
                chunks = self._split_documents(new_docs)
                db.add_documents(chunks)
                db.persist()
        else:
            print("Yeni Chroma veritabanı oluşturuluyor...")
            chunks = self._split_documents(self.documents)
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.db_path
            )
            db.persist()

        print("ChromaDB hazır.")

        # 3️⃣ BM25 + Embedding tabanlı retriever
        chunks = self._split_documents(self.documents)

        basic_retriever = db.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        hybrid_retriever = EnsembleRetriever(
            retrievers=[basic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

        # 4️⃣ MultiQueryRetriever
        if self.use_multiquery:
            retriever_prompt_template = """Sen yardımcı bir asistanısın. Kullanıcının Türkçe sorusunu alıp, benzer dökümanın dilinde arama sorguları üret.
Sadece liste halinde soruları yaz, başka metin yazma.

Soru: {question}

Benzer Arama Sorguları:"""

            retriever_prompt = PromptTemplate(input_variables=["question"], template=retriever_prompt_template)

            self.retriever = MultiQueryRetriever.from_llm(
                retriever=hybrid_retriever,
                llm=self.llm,
                prompt=retriever_prompt
            )
        else:
            self.retriever = hybrid_retriever

        # 5️⃣ Konuşma geçmişi
        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True
        )

        # 6️⃣ Özel prompt
        custom_prompt_template = """Aşağıdaki sohbet geçmişi ve bağlamı kullanarak soruya cevap ver.
Cevabını oluştururken kendi düşünce sürecini yazma.
Güncel bilgilere ulaşmaya çalışma.
Bağlamda yoksa "Üzgünüm, verilen bağlamda bu bilgiye ulaşılamıyor." de.

Sohbet Geçmişi:
{chat_history}

Bağlam:
{context}

Soru:
{question}

Cevap:"""
        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["chat_history", "context", "question"]
        )

        # 7️⃣ RAG zinciri oluşturma
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

        print("RAG zinciri başarıyla oluşturuldu.")

    def _split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(docs)

    def ask(self, query: str) -> Dict:
        """
        Oluşturulan zincir aracılığıyla LLM'e bir soru sorar.
        """
        if not self.qa_chain:
            print("Hata: Pipeline başlatılmadı.")
            return {"result": "Sistem hazır değil."}

        return self.qa_chain.invoke({"question": query})

    def clear_chat_history(self):
        if self.memory:
            self.memory.clear()
            print("Sohbet geçmişi temizlendi.")
