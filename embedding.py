"""
embedding.py
-------------------------------------------------------------------------
Purpose:
    Loads the embedding model that converts text chunks into vectors
    (lists of numbers) capturing their semantic meaning. These vectors
    are what the vector database actually stores and searches over.

Why HuggingFace instead of OpenAI embeddings?
    OpenAI's embedding API needs a paid key. HuggingFace's
    "all-MiniLM-L6-v2" model runs locally/free, is small (~80MB), fast,
    and gives good results for semantic search — same choice made in the
    reference notebook.
-------------------------------------------------------------------------
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a ready-to-use HuggingFace embedding model.

    Note:
        The first time this runs, it will download the model weights
        (~80MB) automatically from HuggingFace. This may take a minute
        depending on your internet connection, but only happens once —
        the model is cached locally after that.

    Returns
    -------
    HuggingFaceEmbeddings
        An embedding model object that can turn text into vectors.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
