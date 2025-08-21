# Uyarıları filtrele
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

# server.py dosyasından RAGServer sınıfını içe aktar
from rag_server import RAGServer

if __name__ == "__main__":
    print("Uygulama başlatılıyor...")
    # RAGServer sınıfından bir örnek oluştur ve çalıştır
    server = RAGServer()
    server.run()
