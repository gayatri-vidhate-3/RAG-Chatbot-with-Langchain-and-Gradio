# 📄 RAG Chatbot — Chat With Any Document (LangChain + Groq + Gradio)

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any document — PDF, TXT, or DOCX — and ask natural-language questions about it. The chatbot answers strictly from the document's own content, not from the LLM's general knowledge, making it grounded, verifiable, and hallucination-resistant by design.

Built as a modular, production-style project rather than a single notebook: each stage of the RAG pipeline (loading, chunking, embedding, storage, retrieval, generation) lives in its own file, wired together through a clean LCEL (LangChain Expression Language) chain.

---

## 🧠 Why this project exists

Most RAG tutorials stop at a notebook. This project takes that same pipeline and restructures it the way a real GenAI application would be shipped:

- Clear separation of concerns (one job per file)
- No hardcoded secrets — API keys loaded from environment variables
- Swappable components (change the embedding model, LLM, or vector store without touching the rest of the app)
- A working UI on top, not just a `print()` statement

It's a compact demonstration of the core RAG skill set: document ingestion, chunking strategy, embeddings, vector search, prompt grounding, and chain composition.

---

## ⚙️ How it works

```
User uploads a document (.pdf / .txt / .docx)
              │
              ▼
   loader_chunking.py
   loads the file, splits it into overlapping text chunks
              │
              ▼
   embedding.py
   converts each chunk into a vector embedding
              │
              ▼
   vector_db.py
   stores the vectors in Chroma, exposes a retriever
              │
              │        User's question
              │              │
              ▼              ▼
   chain.py  (LCEL pipeline)
   retriever finds the most relevant chunks
     → builds a grounded prompt
       → sends it to the Groq LLM
         → parses the response into plain text
              │
              ▼
   Answer shown in the Gradio UI
```

The chain is built with **LCEL** (`retriever | format_docs | prompt | llm | parser`) instead of the older `RetrievalQA` / `create_retrieval_chain` helpers, which were removed/relocated in LangChain 1.0+. LCEL is the current recommended pattern — each step is explicit, composable, and independently testable.

---

## 🚀 Features

- 📂 **Multi-format ingestion** — works with `.pdf`, `.txt`, and `.docx` out of the box
- 🔍 **Semantic search, not keyword search** — retrieves chunks by meaning, using `sentence-transformers` embeddings
- 🧩 **Grounded answers** — the LLM is explicitly prompted to answer only from retrieved context
- 🆓 **Zero paid API dependency for embeddings** — uses a local HuggingFace model; only the LLM call goes to Groq
- ⚡ **Fast inference** — powered by Groq's `llama-3.1-8b-instant`
- 🖥️ **Simple web UI** — built with Gradio, no frontend code needed
- 🔐 **Secrets kept out of source code** — API key loaded from a local `.env` file

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector Store | ChromaDB |
| Document Loaders | LangChain Community (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`) |
| Orchestration | LangChain Expression Language (LCEL) |
| UI | Gradio |
| Config | `python-dotenv` |

---

## 📁 Project Structure

```
rag-chatbot-langchain-groq/
├── main.py             # Entry point — Gradio UI, wires everything together
├── key_loader.py       # Loads GROQ_API_KEY from your .env file
├── LLM.py              # Initializes the Groq LLM
├── embedding.py        # Initializes the HuggingFace embedding model
├── loader_chunking.py  # Loads any file (.pdf/.txt/.docx) + splits into chunks
├── vector_db.py        # Builds the Chroma vector DB + retriever
├── chain.py            # Builds the LCEL RAG chain (retriever + prompt + LLM)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your Groq API key
└── README.md           # This file
```

---

## 🏁 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/rag-chatbot-langchain-groq.git
cd rag-chatbot-langchain-groq
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Copy `.env.example` to `.env` and add your key (get a free one at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=your_actual_groq_key_here
```

### 5. Run the app
```bash
python main.py
```
Open the local URL Gradio prints (usually `http://127.0.0.1:7860`) in your browser.

---

## 💡 Usage

1. Upload a `.pdf`, `.txt`, or `.docx` file and click **Process File**
2. Wait for the "processed successfully" status
3. Ask any question about the document's content
4. Get an answer grounded strictly in what the document actually says

---

## 🔮 Future Improvements

- [ ] Multi-turn conversational memory (follow-up questions)
- [ ] Persist the vector DB to disk so documents don't need re-embedding on every run
- [ ] Show retrieved source chunks alongside each answer for transparency
- [ ] Swap Gradio for a FastAPI backend + custom frontend for production deployment
- [ ] Add support for multiple documents in a single session

---

## 👩‍💻 Author

**Gayatri Vidhate**
Machine Learning Engineer | NLP & GenAI Enthusiast

If you found this useful, feel free to ⭐ the repo or connect with me!
