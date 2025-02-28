from typing import Dict, Any, Optional
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import HuggingFaceHub, Ollama, LlamaCpp, Replicate
from ..models.schema import LLMConfig

class LLMProvider:
    """Provider for LLM initialization."""
    
    @staticmethod
    def initialize_llm(config: LLMConfig) -> Any:
        """Initialize LLM based on configuration."""
        if config.provider == "openai":
            return ChatOpenAI(
                model_name=config.model_name,
                temperature=config.temperature,
                openai_api_key=config.api_key
            )
        elif config.provider == "anthropic":
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                anthropic_api_key=config.api_key
            )
        elif config.provider == "huggingface":
            return HuggingFaceHub(
                repo_id=config.model_name,
                huggingfacehub_api_token=config.api_key,
                **config.additional_params
            )
        elif config.provider == "ollama":
            return Ollama(
                model=config.model_name,
                base_url=config.api_url or "http://localhost:11434",
                **config.additional_params
            )
        elif config.provider == "llamacpp":
            return LlamaCpp(
                model_path=config.model_name,
                **config.additional_params
            )
        elif config.provider == "replicate":
            return Replicate(
                model=config.model_name,
                api_token=config.api_key,
                **config.additional_params
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")