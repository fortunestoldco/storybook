from typing import Dict, Any, Optional
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

@lru_cache()
def get_research_api_keys() -> Dict[str, Optional[str]]:
    """Get all research-related API keys from environment."""
    return {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "tavily": os.getenv("TAVILY_API_KEY"),
        "exa": os.getenv("EXA_API_KEY"),
        "arxiv": os.getenv("ARXIV_API_KEY"),
        "pubmed": os.getenv("PUBMED_API_KEY"),
        "linkup": os.getenv("LINKUP_API_KEY"),
        "serper": os.getenv("SERPER_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "bing": os.getenv("BING_API_KEY")
    }

def get_api_key(api: str) -> Optional[str]:
    """Get specific API key."""
    return get_research_api_keys().get(api)

def validate_api_configuration(config: Dict[str, Any]) -> bool:
    """Validate that required API keys are present."""
    api = config.get("search_api")
    if not api:
        return False
    return bool(get_api_key(api))