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

# Environment variables
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "storybook")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

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

# MongoDB collections
COLLECTION_MANUSCRIPTS = "manuscripts"
COLLECTION_CHARACTERS = "characters"
COLLECTION_WORLDS = "worlds"
COLLECTION_SUBPLOTS = "subplots"
COLLECTION_RESEARCH = "research"
COLLECTION_ANALYSIS = "analysis"

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

