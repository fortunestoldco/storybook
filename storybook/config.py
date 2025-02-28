from __future__ import annotations

# Standard library imports
from typing import Dict, Any, Optional, Union
import os
import logging
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Third-party imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Replicate, LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

__all__ = [
    'LLMProvider',
    'get_llm',
    'create_llm',
    'validate_llm_config',
    'validate_agent_config',
    'DEFAULT_CONFIGS',
    'STATES'
]

logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Create necessary directories
Path("docs").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Default model paths
LLAMACPP_MODEL_PATH = os.getenv("LLAMACPP_MODEL_PATH", "./models/llama-2-7b.Q4_K_M.gguf")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "storybook")
MONGODB_VECTOR_COLLECTION = os.getenv("MONGODB_VECTOR_COLLECTION", "vectors")

# API Configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# LLM Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Research Tools
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# LLM configuration
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_REPLICATE_MODEL = "meta/llama-3-70b-instruct:2a30ae62b32ab1f47530ed5fd32fea38ed408255c747684c41749824a771fa12"

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"

def get_llm(config: Optional[Dict[str, Any]] = None):
    """Get default LLM configuration."""
    use_replicate = config.get("use_replicate", False) if config else False

    return {
        "provider": LLMProvider.REPLICATE if use_replicate else LLMProvider.OPENAI,
        "model": "meta/llama-2-70b-chat" if use_replicate else "gpt-4-turbo-preview",
        "temperature": 0.7
    }

def create_llm(llm_config: Dict[str, Any]) -> BaseChatModel:
    """Create an LLM instance based on configuration."""
    provider = llm_config.get("provider")
    config = llm_config.get("config", {})

    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model_name=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
            streaming=config.get("streaming", False),
        )

    elif provider == LLMProvider.ANTHROPIC:
        return ChatAnthropic(
            model_name=config.get("model_name", "claude-3-sonnet"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
            streaming=config.get("streaming", False),
        )

    elif provider == LLMProvider.REPLICATE:
        return Replicate(
            model=config.get("model_name"),
            temperature=config.get("temperature", 0.7),
            max_new_tokens=config.get("max_new_tokens"),
            streaming=config.get("streaming", False),
        )

    elif provider == LLMProvider.OLLAMA:
        return ChatOllama(
            model=config.get("model_name", "mistral"),
            temperature=config.get("temperature", 0.7),
            num_ctx=config.get("num_ctx"),
            num_gpu=config.get("num_gpu"),
            seed=config.get("seed"),
        )

    elif provider == LLMProvider.LLAMACPP:
        return LlamaCpp(
            model_path=config.get("model_path", LLAMACPP_MODEL_PATH),
            temperature=config.get("temperature", 0.7),
            n_ctx=config.get("n_ctx", 2048),
            n_gpu_layers=config.get("n_gpu_layers", 0),
            streaming=config.get("streaming", False),
        )

    elif provider == LLMProvider.HUGGINGFACE:
        return HuggingFacePipeline(
            model_id=config.get("model_name"),
            task="text-generation",
            temperature=config.get("temperature", 0.7),
            max_new_tokens=config.get("max_new_tokens"),
            trust_remote_code=config.get("trust_remote_code", True),
        )

    elif provider == LLMProvider.CUSTOM:
        return CustomLLM(
            model_path=config.get("model_path"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
            streaming=config.get("streaming", False),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

def validate_llm_config(llm_config: Dict[str, Any]) -> bool:
    """Validate LLM configuration."""
    if not isinstance(llm_config, dict):
        raise ValueError("LLM config must be a dictionary")

    provider = llm_config.get("provider")
    if not provider or provider not in [e.value for e in LLMProvider]:
        raise ValueError(f"Invalid provider: {provider}")

    config = llm_config.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    # Provider-specific validations
    if provider == LLMProvider.LLAMACPP:
        if not config.get("model_path"):
            raise ValueError("model_path is required for LlamaCpp")
    elif provider == LLMProvider.REPLICATE:
        if not config.get("model_name"):
            raise ValueError("model_name is required for Replicate")

    return True

def validate_agent_config(config: Optional[Dict[str, Any]] = None) -> bool:
    """Validate agent configuration."""
    if not config:
        return True  # Default config is valid

    required_fields = ["provider"]
    if not all(field in config for field in required_fields):
        logger.error(f"Missing required fields in agent config: {required_fields}")
        return False

    provider = config.get("provider", "").lower()
    provider_configs = {
        "openai": ["api_key", "model_name"],
        "anthropic": ["api_key", "model_name"],
        "replicate": ["api_token", "model"],
        "llamacpp": ["model_path"],
        "ollama": ["model_name"]
    }

    if provider not in provider_configs:
        logger.error(f"Invalid provider: {provider}")
        return False

    provider_required = provider_configs[provider]
    provider_config = config.get("config", {})

    if not all(field in provider_config for field in provider_required):
        logger.error(f"Missing required fields for {provider}: {provider_required}")
        return False

    return True

# Collection Names
COLLECTION_MANUSCRIPTS = os.getenv("COLLECTION_MANUSCRIPTS", "manuscripts")
COLLECTION_CHARACTERS = os.getenv("COLLECTION_CHARACTERS", "characters")
COLLECTION_WORLDS = os.getenv("COLLECTION_WORLDS", "worlds")
COLLECTION_SUBPLOTS = os.getenv("COLLECTION_SUBPLOTS", "subplots")
COLLECTION_RESEARCH = os.getenv("COLLECTION_RESEARCH", "research")
COLLECTION_ANALYSIS = os.getenv("COLLECTION_ANALYSIS", "analysis")

# Vector Store Configuration
VECTOR_NAMESPACE = f"{MONGODB_DB_NAME}.{MONGODB_VECTOR_COLLECTION}"
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

# Vector Search Configuration
VECTOR_SEARCH_INDEX_DEFAULTS = {
    "name": "vector_search_index",
    "definition": {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "dimensions": 1536,  # OpenAI embeddings dimension
                    "similarity": "cosine",
                    "type": "knnVector",
                },
                "text": {"type": "string"},
                "metadata": {
                    "type": "document"
                }
            }
        }
    }
}

# Collection Vector Search Configurations
COLLECTION_VECTOR_CONFIGS = {
    COLLECTION_MANUSCRIPTS: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "manuscript_vector_index"
    },
    COLLECTION_CHARACTERS: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "character_vector_index"
    },
    COLLECTION_WORLDS: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "world_vector_index"
    },
    COLLECTION_SUBPLOTS: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "subplot_vector_index"
    },
    COLLECTION_RESEARCH: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "research_vector_index"
    },
    COLLECTION_ANALYSIS: {
        **VECTOR_SEARCH_INDEX_DEFAULTS,
        "name": "analysis_vector_index"
    }
}

# Add default configurations
DEFAULT_CONFIGS = {
    "research": {
        "provider": "anthropic",
        "config": {
            "model_name": "claude-3-sonnet",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    },
    "writing": {
        "provider": "openai",
        "config": {
            "model_name": "gpt-4",
            "temperature": 0.9,
            "max_tokens": 4000
        }
    },
    "editorial": {
        "provider": "anthropic",
        "config": {
            "model_name": "claude-3-opus",
            "temperature": 0.3,
            "max_tokens": 4000
        }
    }
}

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

class CustomLLM(BaseChatModel):
    def __init__(self, model_path: str, temperature: float = 0.7, max_tokens: int = 1000, streaming: bool = False):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming

    def generate(self, prompt: str) -> str:
        # Implement the logic to generate text using the custom LLM
        pass

    def stream_generate(self, prompt: str):
        # Implement the logic to stream generate text using the custom LLM
        pass

# Verify configuration loading
if __name__ == "__main__":
    print(f"MongoDB URI: {MONGODB_URI}")
    print(f"MongoDB Database: {MONGODB_DB_NAME}")
