"""
LLM.py
-------------------------------------------------------------------------
Purpose:
    Initializes the Large Language Model (LLM) that generates the final
    answer to the user's question.

Why Groq?
    Groq hosts open-source models (like Meta's LLaMA) on very fast
    inference hardware, and offers a generous free tier — same choice
    made in the reference notebook, instead of the paid OpenAI API.
-------------------------------------------------------------------------
"""

import os
from langchain_groq import ChatGroq

from key_loader import load_groq_key


def get_llm(model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Loads the Groq API key and returns a ready-to-use ChatGroq model.

    Parameters
    ----------
    model_name : str
        Name of the Groq-hosted model to use.
        Default: "llama-3.1-8b-instant" (fast, good general-purpose model,
        same one validated in the reference notebook).

    Returns
    -------
    ChatGroq
        A LangChain-compatible chat model object. It can be called
        directly (llm.invoke("some text")) or plugged into an LCEL chain.
    """
    # Step 1: Get the key from our key_loader module
    groq_key = load_groq_key()

    # Step 2: langchain-groq expects the key to be available as an
    # environment variable named GROQ_API_KEY
    os.environ["GROQ_API_KEY"] = groq_key

    # Step 3: Create and return the chat model
    llm = ChatGroq(model=model_name)
    return llm
