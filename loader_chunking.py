"""
loader_chunking.py
-------------------------------------------------------------------------
Purpose:
    This file does TWO jobs for any file the user uploads:

    1. LOAD  -> Read the raw text out of the file (PDF / TXT / DOCX) and
                convert it into LangChain "Document" objects.
    2. CHUNK -> Split that text into smaller, overlapping pieces so it
                can be embedded and stored effectively in the vector DB.

Why do we need chunking at all?
    LLMs and embedding models work best on small, focused pieces of
    text. If we embedded one giant document as a single vector, the
    retriever could never zoom in on the *specific* paragraph that
    actually answers the user's question. Small overlapping chunks
    solve this.
-------------------------------------------------------------------------
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path: str):
    """
    Detects the file type from its extension and loads it with the
    correct LangChain document loader. This is what makes the app work
    with "any user file".

    Supported types: .pdf, .txt, .docx

    Parameters
    ----------
    file_path : str
        Full path to the file uploaded by the user.

    Returns
    -------
    list[Document]
        Raw (un-chunked) LangChain Document objects.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: '{file_extension}'. "
            f"Supported types are: .pdf, .txt, .docx"
        )

    documents = loader.load()
    return documents


def chunk_documents(documents, chunk_size: int = 300, chunk_overlap: int = 100):
    """
    Splits loaded documents into smaller, overlapping chunks.

    Parameters
    ----------
    documents : list[Document]
        Output of `load_file()`.
    chunk_size : int
        Number of characters in a single chunk.
    chunk_overlap : int
        Number of overlapping characters between two consecutive chunks,
        so a sentence that gets cut at a chunk boundary isn't lost
        entirely from context.

    Returns
    -------
    list[Document]
        Smaller chunked documents, ready to be turned into embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
