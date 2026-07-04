"""
vector_db.py
-------------------------------------------------------------------------
Purpose:
    Builds the Chroma vector database from document chunks, and exposes
    a "retriever" — the object that actually fetches the most relevant
    chunks for a given user question.

Note on the filename:
    You mentioned "vatcor_db.py" in your request — that looks like a
    typo for "vector_db.py", so this file is named correctly as
    "vector_db.py" and imported that way everywhere else in the project.
-------------------------------------------------------------------------
"""

from langchain_community.vectorstores import Chroma


def build_vector_db(chunks, embedding_model) -> Chroma:
    """
    Embeds every chunk using the given embedding model and stores the
    resulting vectors in an in-memory Chroma database.

    Parameters
    ----------
    chunks : list[Document]
        Chunked documents, from loader_chunking.chunk_documents().
    embedding_model : HuggingFaceEmbeddings
        Embedding model from embedding.get_embedding_model().

    Returns
    -------
    Chroma
        A vector store containing the embedded chunks, ready to be
        searched.
    """
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    return db


def get_retriever(db: Chroma):
    """
    Wraps the vector database as a retriever.

    A retriever's only job is: given a question, return the top-matching
    chunks (by semantic similarity) from the vector database.

    Parameters
    ----------
    db : Chroma
        Vector database from build_vector_db().

    Returns
    -------
    VectorStoreRetriever
        Retriever object usable inside an LCEL chain (chain.py uses it).
    """
    return db.as_retriever()
