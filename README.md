# RAG Chatbot — End-to-End Modular Project

Ask questions about any document (PDF / TXT / DOCX) using a Retrieval-Augmented
Generation (RAG) pipeline: LangChain + HuggingFace embeddings + Chroma vector DB +
Groq LLM (`llama-3.1-8b-instant`) + Gradio UI.

This is a modularized, production-style rewrite of the working reference notebook
(`RAG_Chatbot__3_.ipynb`) — same pipeline, same logic, split into clean files.

## Project structure

```
rag_chatbot_app/
├── main.py               # Entry point — Gradio UI, wires everything together
├── key_loader.py          # Loads GROQ_API_KEY from your .env file
├── LLM.py                 # Initializes the Groq LLM
├── embedding.py            # Initializes the HuggingFace embedding model
├── loader_chunking.py     # Loads any file (.pdf/.txt/.docx) + splits into chunks
├── vector_db.py            # Builds the Chroma vector DB + retriever
├── chain.py                # Builds the LCEL RAG chain (retriever + prompt + LLM)
├── requirements.txt        # All dependencies
└── .env.example             # Template for your Groq key
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key**
   - Copy `.env.example` to a new file named `.env`
   - Open `.env` and paste your key:
     ```
     GROQ_API_KEY=your_actual_groq_key_here
     ```
   - Never commit the real `.env` file to GitHub.

3. **Run the app**
   ```bash
   python main.py
   ```
   Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in
   your browser.

## How to use

1. Upload a `.pdf`, `.txt`, or `.docx` file and click **Process File**.
2. Wait for the "processed successfully" status message.
3. Type a question about the document and click **Ask**.
4. The chatbot answers using ONLY the content of your uploaded document.

## Why this structure (for your understanding / interviews)

- **Separation of concerns**: each file has exactly one job (loading, embedding,
  storing, chaining, or serving the UI). This mirrors how RAG systems are
  structured in real production codebases.
- **LCEL instead of `RetrievalQA`**: the chain in `chain.py` is built with the
  `|` pipe-operator style (`retriever | format_docs`, etc.) instead of the older
  `langchain.chains.RetrievalQA` / `create_retrieval_chain` helpers, which were
  removed/relocated in LangChain 1.0+. This is the current recommended pattern.
- **Any file type**: `loader_chunking.py` detects the file extension and picks
  the right LangChain loader (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`),
  so the app isn't limited to PDFs like the original notebook.
- **No hardcoded keys**: `key_loader.py` + `.env` keep your Groq key out of the
  source code entirely — safe to push this project to GitHub.

## Extending this project (ideas for your portfolio README)

- Add chat history / multi-turn conversation memory.
- Swap Chroma for a persistent vector DB (e.g. saved to disk) so you don't
  re-embed the same document every time.
- Add a "sources" panel showing which chunks were used to answer each question.
- Deploy with Docker + FastAPI backend (you already have this experience from
  your SATOC projects) instead of/alongside Gradio.
