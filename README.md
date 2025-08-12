# 📚 Knowledge Scout

**Knowledge Scout** is a Python project that processes data from CSV or TXT files, stores them in a vector database, and answers your questions using **RAG (Retrieval-Augmented Generation)** architecture. It offers a flexible structure with chat history management, custom prompt design, and support for multiple LLMs.

## ⚙️ Features

* **Multi-format data loading** — Support for `.csv` and `.txt` files
* **LangChain integration** for document processing and chunking
* **HuggingFace Embeddings** for vector generation
* **ChromaDB** for persistent vector storage
* **MultiQueryRetriever** for improved search results
* **Gpt4FreeLLM** or **CustomLLM** options
* **Chat history clearing** functionality
* **Fully interactive terminal-based usage**

## 📂 Project Structure

```bash
knowledge-scout/
│
├── data_handler.py     # DataLoader class — Data loading and Document creation
├── rag_system.py       # RagSystem class — RAG pipeline creation and querying
├── llm_model.py        # Gpt4FreeLLM and CustomLLM classes
├── main.py             # Entry point — CLI-based usage
├── requirements.txt    # Dependencies
└── doc/
    └── Top_Anime_data.csv
```

## 🛠 Installation

1. **Clone the project**

```bash
git clone https://github.com/zeynepcagil/knowledge-scout.git
cd knowledge-scout
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **(Optional) Set up OpenAI API key**
   If you want to use `CustomLLM`:

```python
openai_api_token = "YOUR_API_KEY"  # inside main.py
```

## 🚀 Usage

1. **Run the main file**

```bash
python main.py
```

2. **Using with default file**
   The project uses the `doc/Top_Anime_data.csv` file by default.

3. **To use your own data**
   * Change the line in `main.py`:

```python
csv_path = "doc/Top_Anime_data.csv"
```

   to your own `.csv` or `.txt` file path.

4. **Commands**
   * `q` → Exit the program
   * `temizle` → Clear chat history
   * All other inputs are sent as questions to the system.

## 🧩 Components

### **1. DataLoader**
* Reads `.csv` and `.txt` files
* Converts rows or entire text into `Document` objects
* In CSV loading, each row is saved with metadata (headers, row index)

### **2. RagSystem**
* Splits documents into chunks (`RecursiveCharacterTextSplitter`)
* Vectorizes using HuggingFace Embeddings
* Stores in ChromaDB
* Generates richer queries with MultiQueryRetriever
* Includes custom Turkish-English translation-supported search prompt
* Manages chat history with `ConversationBufferWindowMemory`

### **3. LLM Models**
* **Gpt4FreeLLM** — Free model access through g4f library
* **CustomLLM** — OpenAI API compatible request sending

### **4. main.py**
* Manages the flow: Data loading → RAG pipeline initialization → CLI question asking

## 📌 Example Usage

```bash
Enter your question: What is the release year of Naruto?
Answer: 2002
```

```bash
Enter your question: temizle
Chat Geçmiş temizlendi.
```

## 🤝 Contributing

To contribute:
1. Create a fork
2. Create a new branch
3. Commit your changes
4. Send a pull request