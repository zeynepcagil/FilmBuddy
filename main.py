# main.py
from data_handler import DataLoader
from llm_model import  Gpt4FreeLLM
from rag_system import RagSystem
import warnings
from counter import llm_counter

# Gereksiz uyarıları gizle
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

            if "result" in response:
                print(f"Asistan: {response['result']}")
            elif "answer" in response:
                print(f"Asistan: {response['answer']}")
            else:
                print("Asistan: Üzgünüm, yanıtınızı oluştururken bir sorun oluştu.")

            # Her cevap sonrası sayaç değerlerini göster
            print("---")
            print(f"Toplam LLM çağrısı: {llm_counter.call_count}")
            print(
                f"Toplam token kullanımı: {llm_counter.total_tokens} (Giriş: {llm_counter.input_tokens}, Çıkış: {llm_counter.output_tokens})")

    else:
        print("Dokümanlar yüklenemediği için işlem sonlandırıldı.")
