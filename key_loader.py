"""
key_loader.py
-------------------------------------------------------------------------
Purpose:
    Safely loads the GROQ API key needed to talk to the LLM.

Why a separate file?
    Keeping key-loading logic in its own module means:
    - The key is never hardcoded/typed directly into the main code.
    - Any other file (LLM.py, main.py, etc.) can simply import and call
      `load_groq_key()` without worrying about *how* the key is stored.

How the key is supplied (do this yourself, as mentioned):
    1. Create a file named ".env" in this same project folder
       (you can copy ".env.example" and rename it).
    2. Add one line to it:
           GROQ_API_KEY=your_actual_groq_key_here
    3. That's it — this file will automatically pick it up at runtime.
-------------------------------------------------------------------------
"""

import os
from dotenv import load_dotenv


def load_groq_key() -> str:
    """
    Loads the GROQ_API_KEY from the local .env file into the environment
    and returns it as a string.

    Raises
    ------
    ValueError
        If no GROQ_API_KEY is found, so the app fails early with a clear
        message instead of crashing later with a confusing LLM error.
    """
    # load_dotenv() reads the ".env" file (if it exists) and copies its
    # variables into the current process's environment variables.
    load_dotenv()

    groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "Please create a '.env' file in the project folder "
            "(you can copy '.env.example') and add:\n"
            "    GROQ_API_KEY=your_actual_groq_key_here"
        )

    return groq_key
