from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from datetime import datetime
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.llms import Replicate
from langchain_community.chat_models import ChatOllama
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock, ChatBedrockConverse

from storybook.configuration import ModelProvider

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(model_name: str, config: Dict[str, Any] = None) -> BaseChatModel:
    """Load a chat model based on configuration."""
    config = config or {}
    
    # Get model provider
    provider = config.get("provider", "huggingface")
    
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
            model=model_name,  # e.g. "meta/llama-2-70b-chat:latest"
            input={
                "max_length": config.get("max_new_tokens", 512),
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("repetition_penalty", 1.03)
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
    
    elif provider == "llama.cpp":
        model_path = config.get("model_path", os.getenv("LLAMA_CPP_MODEL_PATH"))
        if not model_path:
            raise ValueError("llama_cpp_model_path must be set for llama.cpp provider")
            
        return ChatLlamaCpp(
            model_path=model_path,
            temperature=config.get("temperature", 0.7),
            n_ctx=config.get("context_length", 2048),
            n_gpu_layers=config.get("n_gpu_layers", 8),
            n_batch=config.get("n_batch", 300),
            max_tokens=config.get("max_new_tokens", 512),
            n_threads=multiprocessing.cpu_count() - 1,
            repeat_penalty=config.get("repetition_penalty", 1.1),
            top_p=config.get("top_p", 0.95),
            verbose=config.get("verbose", False)
        )

    elif provider == "openai":
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
        
        if not (aws_access_key_id and aws_secret_access_key and aws_region):
            raise ValueError("AWS credentials and region must be set for Bedrock provider")
            
        # Use ChatBedrockConverse by default since it's recommended for most users
        return ChatBedrockConverse(
            model=model_name,  # e.g. "anthropic.claude-3-sonnet-20240229-v1:0"
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_new_tokens", 512),
            model_kwargs=config.get("model_kwargs", {}),
            credentials_profile_name=config.get("aws_profile"),
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    elif provider == "azure_openai":
        api_key = config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        endpoint = config.get("endpoint", os.getenv("AZURE_OPENAI_ENDPOINT"))
        
        if not (api_key and endpoint):
            raise ValueError("Azure OpenAI API key and endpoint must be set")
            
        return AzureChatOpenAI(
            azure_deployment=config.get("deployment_name", model_name),
            api_version=config.get("api_version", "2023-07-01-preview"),
            azure_endpoint=endpoint,
            api_key=api_key,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_new_tokens", 512),
            model_version=config.get("model_version"),
            streaming=config.get("streaming", False),
            model_kwargs=config.get("model_kwargs", {})
        )
        
    raise ValueError(f"Unsupported model provider: {provider}")


def check_quality_gate(gate_name: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a project meets quality gate requirements to move to the next phase.

    Args:
        gate_name: Name of the quality gate to check.
        metrics: Current quality metrics for the project.
        config: Configuration containing quality gate thresholds.

    Returns:
        Dictionary with gate result information.
    """
    if "quality_gates" not in config or gate_name not in config["quality_gates"]:
        return {"passed": False, "reason": f"Unknown quality gate: {gate_name}"}

    gate = config["quality_gates"][gate_name]
    required_metrics = gate["required_metrics"]
    thresholds = gate["thresholds"]

    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        return {
            "passed": False,
            "reason": f"Missing required metrics: {', '.join(missing_metrics)}"
        }

    failed_thresholds = []
    for metric, threshold in thresholds.items():
        if metrics.get(metric, 0) < threshold:
            failed_thresholds.append(f"{metric} (current: {metrics.get(metric)}, required: {threshold})")

    if failed_thresholds:
        return {
            "passed": False,
            "reason": f"Failed thresholds: {', '.join(failed_thresholds)}"
        }

    return {"passed": True, "timestamp": datetime.now().isoformat()}