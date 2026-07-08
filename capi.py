import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

def get_openrouter_key(custom_key: str = None) -> str:
    """Returns the OpenRouter API Key."""
    if custom_key and custom_key.strip():
        return custom_key.strip()
    return os.getenv("OPENROUTER_API_KEY", "")

def query_openrouter(prompt: str, model: str = "google/gemini-2.5-flash", api_key: str = None) -> str:
    """
    Queries an LLM model on OpenRouter using LangChain's ChatOpenAI wrapper with temperature=0.0.
    """
    key = get_openrouter_key(api_key)
    if not key:
        raise ValueError("OpenRouter API Key is missing. Please set it in .env or enter it in the sidebar.")

    # Initialize LangChain ChatOpenAI client pointing to OpenRouter
    # Set max_tokens=1000 to prevent OpenRouter 402 out-of-credit issues (requests that default to 65k tokens)
    llm = ChatOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=0.0,
        max_tokens=500,
        default_headers={
            "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
            "X-Title": "Resume Tracker RAG App",
        }
    )

    # Invoke LLM via LangChain message interface
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
