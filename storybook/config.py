import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_community.llms import Replicate

# Environment variables
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "storybook")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# LLM configuration
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_REPLICATE_MODEL = "meta/llama-3-70b-instruct:2a30ae62b32ab1f47530ed5fd32fea38ed408255c747684c41749824a771fa12"


def get_llm(
    model: Optional[str] = None, temperature: float = 0.7, use_replicate: bool = False
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
STATES = [
    "START",
    "research",
    "analysis",
    "initialize",
    "character_development",
    "dialogue_enhancement",
    "world_building",
    "subplot_integration",
    "story_arc_evaluation",
    "continuity_check",
    "language_polishing",
    "quality_review",
    "finalize",
    "END",
]
