"""
chain.py
-------------------------------------------------------------------------
Purpose:
    Builds the final RAG (Retrieval-Augmented Generation) chain using
    LangChain Expression Language (LCEL) — the modern `|` pipe-operator
    style of composing steps together.

How the chain works, step by step, when you call rag_chain.invoke(question):

                    question
                        |
                        v
    [1] retriever fetches the most relevant chunks from the vector DB
                        |
                        v
    [2] format_docs() joins those chunks into one plain-text "context" string
                        |
                        v
    [3] prompt template inserts {context} and {input} (the question)
                        |
                        v
    [4] the LLM (Groq) generates a raw answer based on the prompt
                        |
                        v
    [5] StrOutputParser() converts the LLM's raw output into a plain string
                        |
                        v
            final answer string

Why not use `langchain.chains.RetrievalQA` or `create_retrieval_chain`?
    Those older convenience wrappers were removed/relocated in
    LangChain 1.0+. Building the chain manually with LCEL (as done here)
    has no dependency on those legacy modules and is the current
    recommended approach — and it also gives full visibility into each
    step, which is great for learning and interviews.
-------------------------------------------------------------------------
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs) -> str:
    """
    Joins a list of retrieved Document chunks into one plain text block,
    separated by blank lines, so it can be inserted into the prompt as
    a single "context" string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    """
    Assembles and returns the full RAG chain.

    Parameters
    ----------
    retriever : VectorStoreRetriever
        Retriever built from the user's uploaded document
        (vector_db.get_retriever()).
    llm : ChatGroq
        The language model that generates the final answer
        (LLM.get_llm()).

    Returns
    -------
    Runnable
        A ready-to-use chain. Call it like:
            answer = rag_chain.invoke("your question here")
    """
    # The prompt tells the LLM to answer ONLY using the retrieved
    # context, which keeps answers grounded in the uploaded document
    # instead of the LLM's own general knowledge.

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context below:\n\n"
        "{context}\n\n"
        "Question: {input}"
    )

    # LCEL pipeline: LCEL = LangChain Expression Language. It's LangChain's syntax for chaining components together using the | (pipe) operator 
    #                       — the same idea as Unix pipes, where the output of one step becomes the input of the next.

    # - The dict below runs both branches on the same input question:
    #     "context" -> passes the question through the retriever, then
    #                  formats the returned chunks into plain text
    #     "input"   -> passes the raw question straight through unchanged

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
