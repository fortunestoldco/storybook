from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper  # Or other tool dependencies
from typing import Dict, Any, Optional

#from .config import config #Assumes a .config file exists.

# Example: Define a search tool (if needed)
def get_search_results(query: str) -> str:
    """
    Fetches search results for a given query using SerpAPI.
    """
    #if not config.ENABLE_SEARCH:
    #    return "Search is disabled."
    #Ensure you have a valid API key in your environemnt variables to use the tool.
    search = SerpAPIWrapper()
    return search.run(query)


# Define other tools as needed (e.g., writing assistant, image generation, etc.)
def generate_chapter_outline(story_summary: str) -> str:
    """
    Generates a chapter outline based on a story summary.  Placeholder implementation.
    """
    return f"Placeholder chapter outline for: {story_summary}"

def update_story_bible(story_bible: str, new_information: str) -> str:
    """
    Updates the story bible with new information and maintains consistency.
    
    Args:
        story_bible (str): The current story bible content
        new_information (str): New information to be added to the story bible
    
    Returns:
        str: Updated story bible content
    """
    if not story_bible:
        return new_information
    
    # Combine existing story bible with new information
    updated_bible = f"{story_bible}\n\n--- Updated Information ---\n{new_information}"
    
    # Here you could add more sophisticated merging logic:
    # - Check for contradictions
    # - Organize by categories
    # - Remove duplicates
    # - etc.
    
    return updated_bible


# Create Tool instances
search_tool = Tool(
    name="search",
    func=get_search_results,
    description="useful for when you need to answer questions about current events or the current state of the world. input should be a search query.",
)

chapter_outline_tool = Tool(
    name="chapter_outline_generator",
    func=generate_chapter_outline,
    description="Generates an outline of chapters for a story based on a brief summary. Good for developing the plot. Input should be a brief summary of what you would like your plot to focus on."
)

story_bible_tool = Tool(
    name="story_bible_updater",
    func=update_story_bible,
    description="Updates the story bible with new information while maintaining consistency. Input should be a dictionary containing 'story_bible' and 'new_information'."
)

# Define the list of available tools
available_tools = [search_tool, chapter_outline_tool, story_bible_tool]

def get_tools():
    """Returns available tools."""
    return available_tools

# Export the update_story_bible function
__all__ = ['get_tools', 'update_story_bible']
