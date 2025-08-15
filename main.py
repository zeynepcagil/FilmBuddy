# main.py
from data_handler import DataLoader
from llm_model import CustomLLM, Gpt4FreeLLM
from rag_system import RagSystem
import warnings

# Generate content
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

if __name__ == "__main__":
    csv_path = "doc/n_movies.csv"

    openai_api_token = "Your API Token"
    data_loader = DataLoader(csv_path)
    documents = data_loader.load_data()

    if documents:

        llm = Gpt4FreeLLM()

        rag_system = RagSystem(documents=documents, llm=llm)
        rag_system.initialize_pipeline()

        while True:
            query = input("Sorunuzu giriniz: ")
            if query.lower() == "q":
                print("Uygulamadan çıkılıyor.")
                break
            elif query.lower() == "temizle":
                rag_system.clear_chat_history()
                continue

            response = rag_system.ask(query)

            if "answer" in response:
                print(f"Cevap: {response['answer']}")
            else:
                print(f"Yanıt alınamadı. Yanıt içeriği: {response}")

    else:
        print("Dokümanlar yüklenemediği için işlem sonlandırıldı.")
