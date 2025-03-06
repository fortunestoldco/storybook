from storybook.configuration import Configuration
from storybook.graph import get_storybook_supervisor, get_storybook_initialization, get_storybook_development, get_storybook_creation, get_storybook_refinement, get_storybook_finalization
import os
from dotenv import load_dotenv
from langsmith import Client

__all__ = [
    "Configuration", 
    "get_storybook_supervisor", 
    "get_storybook_initialization", 
    "get_storybook_development", 
    "get_storybook_creation", 
    "get_storybook_refinement", 
    "get_storybook_finalization"
]

def initialize_langsmith():
    load_dotenv()
    
    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found")
        
    # Initialize LangSmith client with explicit API key
    client = Client(
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=os.getenv("LANGCHAIN_API_KEY"),
    )
    return client
