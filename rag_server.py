import warnings
from flask import Flask, request, jsonify,render_template
from threading import Thread
# Diğer modüllerinizi import edin
from data_handler import DataLoader
from llm_model import Gpt4FreeLLM
from rag_system import RagSystem
from counter import llm_counter


class RAGServer:
    """
    RAG sistemini bir HTTP sunucusu olarak yönetmek için sınıf.
    """
    def __init__(self, csv_path="doc/n_movies.csv"):
        """
        Flask uygulamasını ve RAG sistemini başlatır.
        """
        self.app = Flask(__name__)
        self.rag_system = self._initialize_rag_system(csv_path)
        self._setup_routes()

    def _initialize_rag_system(self, csv_path):
        """
        RAG sistemini kurar ve dokümanları yükler.
        """
        data_loader = DataLoader(csv_path)
        documents = data_loader.load_data()

        if not documents:
            print("Dokümanlar yüklenemediği için işlem sonlandırıldı.")
            exit()

        llm = Gpt4FreeLLM()
        rag_system = RagSystem(documents=documents, llm=llm)
        rag_system.initialize_pipeline()
        return rag_system

    def _setup_routes(self):
        """
        HTTP end-point'lerini tanımlar.
        """
        @self.app.route("/")
        def index():
            return render_template('index.html')

        @self.app.route("/ask", methods=["POST"])
        def ask_endpoint():
            data = request.get_json()
            if not data or "query" not in data:
                return jsonify({"error": "JSON gövdesi bulunamadı veya 'query' parametresi eksik."}), 400

            query = data["query"]

            try:
                response_dict = self.rag_system.ask(query)
                return jsonify({
                    "query": query,
                    "response": response_dict["result"],
                    "llm_calls": llm_counter.call_count,
                    "tokens": {
                        "total": llm_counter.total_tokens,
                        "input": llm_counter.input_tokens,
                        "output": llm_counter.output_tokens
                    }
                })
            except Exception as e:
                return jsonify({"error": f"Sorgu işlenirken bir hata oluştu: {str(e)}"}), 500

    def terminal_loop(self):
        """
        Terminal üzerinden interaktif sorgu döngüsü.
        """
        print("Sistem kullanıma hazır. Çıkmak için 'q', sohbet geçmişini temizlemek için 'temizle' yazın.")
        print("---")
        while True:
            query = input("Sorunuzu giriniz: ")
            if query.lower() == "q":
                print("Uygulamadan çıkılıyor.")
                break
            elif query.lower() == "temizle":
                self.rag_system.clear_chat_history()
                print("Sohbet geçmişi temizlendi.")
                continue

            response_dict = self.rag_system.ask(query)
            print(f"Cevap: {response_dict['result']}")
            print("---")
            print(f"Toplam LLM çağrısı: {llm_counter.call_count}")
            print(f"Toplam token kullanımı: {llm_counter.total_tokens} "
                  f"(Giriş: {llm_counter.input_tokens}, Çıkış: {llm_counter.output_tokens})")
            print("---")

    def run(self):
        """
        HTTP sunucusunu ve terminal döngüsünü başlatır.
        """
        Thread(target=self.terminal_loop, daemon=True).start()
        print("HTTP sunucusu başlatılıyor...")

        self.app.run(host="0.0.0.0", port=5000, debug=False)
