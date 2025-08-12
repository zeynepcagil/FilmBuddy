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


class RagSystem:
    """
    Veri yükleme, embedding, vektör veritabanı ve LLM zincirini yöneten ana RAG sınıfı.
    """

    def __init__(self, documents: List[Document], llm: LLM):
        self.documents = documents
        self.llm = llm
        self.qa_chain = None
        self.retriever = None
        self.memory = None

    def initialize_pipeline(self):
        """
        RAG pipeline'ını oluşturur ve başlatır.
        """
        print("Dokümanlar işleniyor ve parçalara ayrılıyor...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.documents)

        print("Embedding modeli yükleniyor...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("Dokümanlar ChromaDB'ye ekleniyor...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db_bge_csv"
        )
        print("ChromaDB hazır ve dokümanlarla dolduruldu.")

        # MultiQueryRetriever için özel prompt'u tanımlıyoruz.
        retriever_prompt_template = """Sen yardımcı bir asistanısın. Kullanıcının Türkçe sorusunu alıp, benzer ENGLISH arama sorguları üret.

Sadece liste halinde soruları yaz, başka metin yazma.

Soru: {question}

Benzer Arama Sorguları:"""

        retriever_prompt = PromptTemplate(input_variables=["question"], template=retriever_prompt_template)

        basic_retriever = db.as_retriever(search_kwargs={"k": 5})
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=basic_retriever,
            llm=self.llm,
            prompt=retriever_prompt
        )


        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True
        )


        custom_prompt_template = """Aşağıdaki sohbet geçmişi ve bağlamı kullanarak soruya cevap ver.
Cevabını oluştururken kendi düşünce sürecini, analizlerini veya adım adım ilerleyişini yazma.
İnternette **asla** arama yapma veya güncel bilgilere ulaşmaya çalışma.
Cevap için gerekli bilgi bağlamda kesinlikle bulunmuyorsa, sadece "Üzgünüm, verilen bağlamda bu bilgiye ulaşılamıyor." şeklinde yanıt ver.
Sadece bağlamda yer alan bilgilere dayanarak, doğrudan ve kısa bir şekilde yanıtla.

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

        # 3. ConversationalRetrievalChain'i Özel Prompt ile Oluştur
        # chain_type_kwargs içinde custom prompt'u belirtiyoruz
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        print("RAG zinciri ve özel prompt başarıyla oluşturuldu.")

    def ask(self, query: str) -> Dict:
        """
        Oluşturulan zincir aracılığıyla LLM'e bir soru sorar.
        """
        if not self.qa_chain:
            print("Hata: RAG boru hattı başlatılmadı. Önce initialize_pipeline() metodunu çağırın.")
            return {"result": "Sistem hazır değil."}

        # ConversationalRetrievalChain'e 'question' parametresiyle soruyu gönder
        return self.qa_chain.invoke({"question": query})

    # rag_pipeline.py dosyasındaki RagSystem sınıfına ekleyin

    def clear_chat_history(self):
        """
        Sohbet geçmişini tamamen siler.
        """
        if self.memory:
            self.memory.clear()
            print("Sohbet geçmişi temizlendi.")
        else:
            print("Bellek nesnesi başlatılmamış.")