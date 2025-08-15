from typing import List
import pandas as pd
from langchain.docstore.document import Document



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
        Şu anda CSV, TXT ve PDF desteklenmektedir.
        """
        if self.file_type == 'csv':
            return self._load_csv()
        elif self.file_type == 'txt':
            return self._load_txt()
        elif self.file_type == 'pdf':
            return self._load_pdf()
        else:
            print(f"Hata: Desteklenmeyen dosya türü: {self.file_type}")
            return []

    def _load_txt(self) -> List[Document]:
        """
        TXT dosyasından veriyi yükler.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            document = Document(page_content=text_content, metadata={"source": self.file_path})
            print(f"'{self.file_path}' dosyasından TXT olarak yüklendi.")
            return [document]
        except Exception as e:
            print(f"TXT dosyasından veri okunurken bir hata oluştu: {e}")
            return []

    def _load_pdf(self) -> List[Document]:
        """
        PDF dosyasını pdfplumber ile yükler ve tüm sayfaları birleştirerek tek Document olarak döner.
        """
        try:
            import pdfplumber

            full_text = ""
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

            document = Document(
                page_content=full_text,
                metadata={"source": self.file_path}
            )
            print(f"'{self.file_path}' PDF dosyası pdfplumber ile başarıyla yüklendi.")
            return [document]

        except Exception as e:
            print(f"PDF dosyasından veri okunurken hata oluştu: {e}")
            return []

    def _load_csv(self) -> List[Document]:
        """
        CSV dosyasını parça parça yükler.
        """
        try:
            documents = []
            chunk_size = 800
            print(f"'{self.file_path}' dosyasını {chunk_size} satırlık parçalar halinde yüklüyor...")

            df_iterator = pd.read_csv(self.file_path, chunksize=chunk_size)
            first_chunk = next(df_iterator)
            headers = list(first_chunk.columns)
            headers_string = ", ".join(headers)

            for index, row in first_chunk.iterrows():
                text_content = " ".join(
                    f"{header}: {value}" for header, value in zip(headers, row.values) if pd.notna(value)
                )
                metadata = {
                    "source": self.file_path,
                    "row_index": index,
                    "headers": headers_string
                }
                documents.append(Document(page_content=text_content, metadata=metadata))

            for chunk in df_iterator:
                for index, row in chunk.iterrows():
                    text_content = " ".join(
                        f"{headers[i]}: {row.iloc[i]}" for i in range(len(headers)) if pd.notna(row.iloc[i])
                    )
                    metadata = {
                        "source": self.file_path,
                        "row_index": index,
                        "headers": headers_string
                    }
                    documents.append(Document(page_content=text_content, metadata=metadata))

            print(f"'{self.file_path}' dosyasından toplam {len(documents)} CSV satırı yüklendi.")
            return documents
        except Exception as e:
            print(f"CSV dosyasından veri okunurken bir hata oluştu: {e}")
            return []
