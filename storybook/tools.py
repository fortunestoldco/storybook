"""Basic example of using tools."""
from typing import List

from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
    
def get_tools() -> List:
    """Get tools."""
    tools = [
        get_word_length, TavilySearchResults(max_results=1)
    ]
    return tools
