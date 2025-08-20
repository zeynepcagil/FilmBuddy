import warnings
from data_handler import DataLoader
from llm_model import Gpt4FreeLLM
from rag_system import RagSystem
from counter import llm_counter

# LangChain ve Hugging Face uyarılarını filtrele
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

if __name__ == "__main__":
    csv_path = "doc/n_movies.csv"

    data_loader = DataLoader(csv_path)
    documents = data_loader.load_data()

    if not documents:
        print("Dokümanlar yüklenemediği için işlem sonlandırıldı.")
        exit()

    llm = Gpt4FreeLLM()
    rag_system = RagSystem(documents=documents, llm=llm)
    rag_system.initialize_pipeline()

    print("Sistem kullanıma hazır. Çıkmak için 'q', sohbet geçmişini temizlemek için 'temizle' yazın.")
    print("---")

    while True:
        query = input("Sorunuzu giriniz: ")

        if query.lower() == "q":
            print("Uygulamadan çıkılıyor.")
            break
        elif query.lower() == "temizle":
            rag_system.clear_chat_history()
            print("Sohbet geçmişi temizlendi.")
            continue

        # RagSystem'in ana sorgu işleme metodunu çağırıyoruz
        response_dict = rag_system.ask(query)
        print(f"Cevap: {response_dict['result']}")

        print("---")
        print(f"Toplam LLM çağrısı: {llm_counter.call_count}")
        print(f"Toplam token kullanımı: {llm_counter.total_tokens} "
              f"(Giriş: {llm_counter.input_tokens}, Çıkış: {llm_counter.output_tokens})")
        print("---")