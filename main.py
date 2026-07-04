"""
main.py
-------------------------------------------------------------------------
Purpose:
    This is the ENTRY POINT of the whole application. Run this file to
    launch the RAG chatbot's web interface (built with Gradio).

End-to-end flow:
    1. User uploads any file (.pdf / .txt / .docx) through the UI.
    2. User clicks "Process File":
         load_file()        -> reads raw text from the file
         chunk_documents()  -> splits text into small overlapping chunks
         build_vector_db()  -> embeds chunks and stores them in Chroma
         get_retriever()    -> wraps the DB as a retriever
         
         build_rag_chain()  -> wires retriever + LLM into one pipeline
    3. User types a question and clicks "Ask":
         rag_chain.invoke(question) -> returns an answer grounded ONLY
                                        in the uploaded document's content

How to run:
    1. Make sure requirements.txt is installed:
           pip install -r requirements.txt
    2. Make sure your ".env" file has GROQ_API_KEY set
       (see .env.example).
    3. Run:
           python main.py
    4. Open the local URL Gradio prints in your terminal.
-------------------------------------------------------------------------
"""

import gradio as gr

from loader_chunking import load_file, chunk_documents
from embedding import get_embedding_model
from vector_db import build_vector_db, get_retriever
from LLM import get_llm
from chain import build_rag_chain

# -------------------------------------------------------------------
# These are loaded ONCE when the app starts (not on every question),
# so the app stays fast and doesn't reload the embedding model / LLM
# every single time a user asks something.
# -------------------------------------------------------------------
embedding_model = get_embedding_model()
llm = get_llm()


def process_file(file):
    """
    Runs the full ingestion pipeline for a newly uploaded file and
    builds a RAG chain specific to that document.

    Parameters
    ----------
    file : tempfile-like object
        The file object Gradio provides after upload. `file.name`
        holds the path to the uploaded file on disk.

    Returns
    -------
    tuple(Runnable or None, str)
        The built RAG chain (to be stored in Gradio state) and a status
        message to display to the user.
    """
    if file is None:
        return None, "Please upload a file first."

    file_path = file.name

    try:
        # Step 1: Load the raw text from the file
        documents = load_file(file_path)

        # Step 2: Split the text into overlapping chunks
        chunks = chunk_documents(documents)

        # Step 3: Embed the chunks and store them in a vector database
        db = build_vector_db(chunks, embedding_model)

        # Step 4: Turn the vector database into a retriever
        retriever = get_retriever(db)

        # Step 5: Build the final RAG chain (retriever + prompt + LLM)
        rag_chain = build_rag_chain(retriever, llm)

    except Exception as error:
        # Surface any loading/embedding errors clearly in the UI instead
        # of crashing the whole app.
        return None, f"Failed to process file: {error}"

    file_name = file_path.split("/")[-1].split("\\")[-1]
    status_message = f"'{file_name}' processed successfully! You can now ask questions below."
    return rag_chain, status_message


def answer_question(question, rag_chain):
    """
    Answers a user's question using the RAG chain built for the
    currently uploaded document.

    Parameters
    ----------
    question : str
        The question typed by the user.
    rag_chain : Runnable or None
        The chain built by process_file(), stored in Gradio state.

    Returns
    -------
    str
        The generated answer, or a helpful message if something is
        missing.
    """
    if rag_chain is None:
        return "Please upload and process a file first."

    if not question or question.strip() == "":
        return "Please enter a question."

    try:
        response = rag_chain.invoke(question)
    except Exception as error:
        return f"Something went wrong while generating the answer: {error}"

    return response


# -------------------------------------------------------------------
# Gradio UI layout
# -------------------------------------------------------------------
with gr.Blocks(title="RAG Chatbot - Ask Questions About Any Document") as app:

    gr.Markdown("# 📄 RAG Chatbot")
    gr.Markdown(
        "Upload any document (**.pdf / .txt / .docx**) and ask questions "
        "about it. The chatbot answers strictly using the content of "
        "your uploaded document."
    )

    # gr.State keeps a Python object (our rag_chain) alive across
    # multiple button clicks, scoped to a single user's browser session.
    rag_chain_state = gr.State(value=None)

    with gr.Row():
        file_input = gr.File(
            label="Upload your document (.pdf, .txt, .docx)",
            file_types=[".pdf", ".txt", ".docx"],
        )
        process_button = gr.Button("Process File", variant="primary")

    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a question about your document",
            placeholder="e.g. What are the key skills mentioned in this document?",
        )
        ask_button = gr.Button("Ask", variant="primary")

    answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)

    # ---------------------------------------------------------------
    # Wiring: connect UI buttons to the Python functions above
    # ---------------------------------------------------------------
    process_button.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[rag_chain_state, status_output],
    )

    ask_button.click(
        fn=answer_question,
        inputs=[question_input, rag_chain_state],
        outputs=[answer_output],
    )


if __name__ == "__main__":
    # launch() starts a local web server and prints a URL you can open
    # in your browser. Set share=True if you want a temporary public link.
    app.launch()
