from __future__ import annotations

# Standard library imports
from typing import Any, Optional
import os

# Third-party imports
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "storybook")

# API Configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Research Tools
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# LLM configuration
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_REPLICATE_MODEL = "meta/llama-3-70b-instruct:2a30ae62b32ab1f47530ed5fd32fea38ed408255c747684c41749824a771fa12"

def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    use_replicate: bool = False,
):
    """Get a configured LLM instance."""
    if use_replicate:
        return Replicate(
            model=model or DEFAULT_REPLICATE_MODEL,
            temperature=temperature,
            api_key=REPLICATE_API_TOKEN,
        )
    else:
        return ChatOpenAI(
            model=model or DEFAULT_OPENAI_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )

# Collection Names
COLLECTION_MANUSCRIPTS = os.getenv("COLLECTION_MANUSCRIPTS", "manuscripts")
COLLECTION_CHARACTERS = os.getenv("COLLECTION_CHARACTERS", "characters")
COLLECTION_WORLDS = os.getenv("COLLECTION_WORLDS", "worlds")
COLLECTION_SUBPLOTS = os.getenv("COLLECTION_SUBPLOTS", "subplots")
COLLECTION_RESEARCH = os.getenv("COLLECTION_RESEARCH", "research")
COLLECTION_ANALYSIS = os.getenv("COLLECTION_ANALYSIS", "analysis")

# Define the states for our state machine
STATES = {
    "START": "start",
    "RESEARCH": "research",
    "ANALYSIS": "analysis",
    "INITIALIZE": "initialize",
    "CHARACTER_DEVELOPMENT": "character_development",
    "DIALOGUE_ENHANCEMENT": "dialogue_enhancement",
    "WORLD_BUILDING": "world_building",
    "SUBPLOT_INTEGRATION": "subplot_integration",
    "STORY_ARC_EVALUATION": "story_arc_evaluation",
    "CONTINUITY_CHECK": "continuity_check",
    "LANGUAGE_POLISHING": "language_polishing",
    "QUALITY_REVIEW": "quality_review",
    "FINALIZE": "finalize",
    "END": "end",
}
