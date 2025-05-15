
# 🧠 LangChain RAG Assistant with Ollama

This is a Retrieval-Augmented Generation (RAG) chatbot using [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com), and local `.txt` documents. It performs semantic search over a knowledge base and generates accurate answers using a local LLM.

---

## ✨ Features

- 🔍 Document loading and recursive text splitting
- 🧠 Semantic vector search with FAISS
- 🗣️ LLM response generation using Ollama (`mistral:instruct`)
- 📚 Easy knowledge base updates via the `knowledge_base/` folder
- ⚡ Efficient RAG chain setup with LangChain

---

## 🧩 Tech Stack

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- FAISS (Vector DB)
- Mistral or any Ollama-compatible model
- Local embeddings model (`nomic-embed-text` via Ollama)

---

## 📦 Installation

Install dependencies:

```bash
pip install langchain langchain-community langchain-core
````

Install FAISS if needed:

```bash
pip install faiss-cpu
```

---

## 🔧 Setup

1. **Start Ollama**:

   ```bash
   ollama serve
   ```

2. **Pull models**:

   ```bash
   ollama pull mistral:instruct
   ollama pull nomic-embed-text
   ```

3. **Prepare documents**:

   * Put your `.txt` files inside the `knowledge_base/` directory.
   * If the folder is empty, a placeholder file will be created.

---

## 🚀 Running the App

```bash
python rag_assistant.py
```

You’ll see:

```text
Text-based RAG assistant is ready. Type 'exit' to quit.
You:
```

Start chatting! Use `"clear index"` to regenerate the vector store after updating documents.

---

## 🧠 How It Works

1. **Load Documents** – Reads all `.txt` files in `knowledge_base/`
2. **Text Splitter** – Chunks documents into 1000-character segments with 200 overlap
3. **Embedding** – Uses `nomic-embed-text` to vectorize chunks
4. **Vector Store** – Stores embeddings in FAISS (auto-persistent)
5. **Retrieval** – Top-k similar chunks fetched per query
6. **Prompt** – Context + question passed to Ollama for generation

---

## 📂 Project Structure

```
rag_assistant.py            # Main script
knowledge_base/             # Folder for your .txt files
faiss_index_store/          # Auto-created vector DB directory
README.md                   # This file
```

---

## 📝 Example Usage

```
You: What is LangChain?
Bot: LangChain is a framework for developing applications powered by language models...
```

---

## 🧼 Commands

* `exit` or `quit` – Terminates the app
* `clear index` – Deletes and rebuilds the vector store from updated documents

---

## 🛠️ Customization

* To change chunk size or overlap, edit:

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

* Switch model in config:

```python
LLM_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "nomic-embed-text"
```

---

## ❗ Troubleshooting

* Make sure `ollama` is running: `ollama serve`
* Ensure required models are pulled
* Ensure documents exist in `knowledge_base/`
* Run `clear index` if documents were changed

---

## 📄 License

MIT License

```

Let me know if you'd like to add voice input/output or deploy as a web app.
```
