from typing import List

import pandas as pd
from langchain.docstore.document import Document


# Veri yükleme ve işleme için bir sınıf
class DataLoader:
    """
    Farklı dosya türlerinden veri yükleme ve LangChain Document'larına dönüştürme sınıfı.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1].lower()

    def load_data(self) -> List[Document]:
        """
        Dosya türüne göre uygun yükleme fonksiyonunu çağırır.
        Şu anda CSV ve TXT desteklenmektedir.
        """
        if self.file_type == 'csv':
            return self._load_csv()
        elif self.file_type == 'txt':
            return self._load_txt()
        else:
            print(f"Hata: Desteklenmeyen dosya türü: {self.file_type}")
            return []


    def _load_txt(self) -> List[Document]:
        """
        Bir TXT dosyasından veriyi satır satır yükler. Tüm içeriği tek bir
        LangChain Document'a dönüştürür.
        """
        try:
            text_content = ""
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Dosyayı satır satır okur, böylece büyük dosyalar için belleği yormaz
                for line in f:
                    text_content += line

            document = Document(page_content=text_content, metadata={"source": self.file_path})
            print(f"'{self.file_path}' dosyasından tek bir doküman başarıyla yüklendi.")
            return [document]
        except Exception as e:
            print(f"TXT dosyasından veri okunurken bir hata oluştu: {e}")
            return []

    # data_handler.py dosyasındaki güncellenmiş _load_csv metodu

    def _load_csv(self) -> List[Document]:
        """
        Bir CSV dosyasından veriyi parça parça yükler, her parçayı bir LangChain Document'a dönüştürür.
        Bu versiyon, başlıkları da metadata olarak ekler.
        """
        try:
            documents = []
            chunk_size = 800
            print(f"'{self.file_path}' dosyasını {chunk_size} satırlık parçalar halinde yüklüyor...")

            # Dosyanın başlıklarını almak için ilk parçayı okuyun
            df_iterator = pd.read_csv(self.file_path, chunksize=chunk_size)
            first_chunk = next(df_iterator)
            headers = list(first_chunk.columns)

            # Hata veren kısmı düzeltme: Başlıklar listesini bir stringe dönüştür
            headers_string = ", ".join(headers)

            # İlk parçayı da işleyin
            for index, row in first_chunk.iterrows():
                # Metin içeriği için başlıkları ve verileri birleştirin
                text_content = " ".join(
                    f"{header}: {value}" for header, value in zip(headers, row.values) if pd.notna(value))

                # Metadata'yı oluşturun, başlıkları string olarak ekleyin
                metadata = {
                    "source": self.file_path,
                    "row_index": index,
                    "headers": headers_string  # Burayı güncelledik
                }
                documents.append(Document(page_content=text_content, metadata=metadata))

            # Kalan parçaları işleyin
            for chunk in df_iterator:
                for index, row in chunk.iterrows():
                    text_content = " ".join(
                        f"{headers[i]}: {row.iloc[i]}" for i in range(len(headers)) if pd.notna(row.iloc[i]))
                    metadata = {
                        "source": self.file_path,
                        "row_index": index,
                        "headers": headers_string  # Burayı güncelledik
                    }
                    documents.append(Document(page_content=text_content, metadata=metadata))

            print(f"'{self.file_path}' dosyasından toplam {len(documents)} doküman başarıyla yüklendi.")
            return documents
        except Exception as e:
            print(f"CSV dosyasından veri okunurken bir hata oluştu: {e}")
            return []
