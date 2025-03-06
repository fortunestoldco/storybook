from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
import os
import multiprocessing
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.llms import Replicate
from langchain_community.chat_models import ChatOllama, ChatLlamaCpp
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock, ChatBedrockConverse

def load_chat_model(model_name: str, config: Dict[str, Any] = None) -> BaseChatModel:
    """Load a chat model based on configuration."""
    config = config or {}
    
    # Get model provider
    provider = config.get("provider", "openai")
    
    if provider == "huggingface":
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task=config.get("task", "text-generation"),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.7),
            repetition_penalty=config.get("repetition_penalty", 1.03),
            api_key=config.get("api_key", os.getenv("HUGGINGFACE_API_KEY"))
        )
        return ChatHuggingFace(llm=llm)
        
    elif provider == "replicate":
        return Replicate(
            model=model_name,
            input={
                "max_length": config.get("max_new_tokens", 512),
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.95)
            },
            api_key=config.get("api_key", os.getenv("REPLICATE_API_KEY"))
        )
        
    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=config.get("temperature", 0.7),
            base_url=config.get("base_url", os.getenv("OLLAMA_BASE_URL")),
            repeat_penalty=config.get("repetition_penalty", 1.03),
            num_ctx=config.get("max_new_tokens", 512)
        )
        
    elif provider == "anthropic":
        api_key = config.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError("anthropic_api_key must be set for Anthropic provider")
            
        return ChatAnthropic(
            model=model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_new_tokens", 512),
            anthropic_api_key=api_key
        )
        
    elif provider == "bedrock":
        aws_access_key_id = config.get("aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_access_key = config.get("aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        aws_region = config.get("aws_region", os.getenv("AWS_REGION"))
        
        if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
            raise ValueError("AWS credentials must be set for Bedrock provider")
            
        return ChatBedrockConverse(
            model_id=model_name,
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_kwargs={
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_new_tokens", 512)
            }
        )
        
    # Default to OpenAI
    api_key = config.get("api_key", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise ValueError("openai_api_key must be set for OpenAI provider")
        
    return ChatOpenAI(
        model=model_name,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_new_tokens", 512),
        model_kwargs=config.get("model_kwargs", {}),
        openai_api_key=api_key
    )