from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from datetime import datetime
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.llms import Replicate, Ollama
from langchain.chat_models import ChatOllama
from storybook.configuration import ModelProvider, Configuration
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock, ChatBedrockConverse


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


def load_chat_model(agent_name: str, config: Configuration) -> BaseChatModel:
    """Load a chat model for a specific agent."""
    
    # Get agent-specific config or fall back to default
    model_config = config.agent_model_configs.get(
        agent_name, 
        config.default_model_config
    )
    
    provider = model_config.get("provider", config.model_provider)
    
    if provider == ModelProvider.HUGGINGFACE:
        llm = HuggingFaceEndpoint(
            repo_id=model_config["model_name"],
            task=model_config.get("task", "text-generation"),
            max_new_tokens=model_config.get("max_new_tokens", 512),
            temperature=model_config.get("temperature", 0.7),
            repetition_penalty=model_config.get("repetition_penalty", 1.03),
            api_key=config.huggingface_api_key
        )
        return ChatHuggingFace(llm=llm)
        
    elif provider == ModelProvider.REPLICATE:
        return Replicate(
            model=model_config["model_name"],  # e.g. "meta/llama-2-70b-chat:latest"
            input={
                "max_length": model_config.get("max_new_tokens", 512),
                "temperature": model_config.get("temperature", 0.7),
                "top_p": model_config.get("repetition_penalty", 1.03)
            },
            api_key=config.replicate_api_key
        )
        
    elif provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config["model_name"],
            temperature=model_config.get("temperature", 0.7),
            base_url=config.ollama_base_url,
            repeat_penalty=model_config.get("repetition_penalty", 1.03),
            num_ctx=model_config.get("max_new_tokens", 512)
        )
        
    # Add llama.cpp handling
    elif provider == ModelProvider.LLAMA_CPP:
        if not config.llama_cpp_model_path:
            raise ValueError("llama_cpp_model_path must be set for llama.cpp provider")
            
        return ChatLlamaCpp(
            model_path=config.llama_cpp_model_path,
            temperature=model_config.get("temperature", 0.7),
            n_ctx=model_config.get("context_length", 2048),
            n_gpu_layers=model_config.get("n_gpu_layers", 8),
            n_batch=model_config.get("n_batch", 300),
            max_tokens=model_config.get("max_new_tokens", 512),
            n_threads=multiprocessing.cpu_count() - 1,
            repeat_penalty=model_config.get("repetition_penalty", 1.1),
            top_p=model_config.get("top_p", 0.95),
            verbose=model_config.get("verbose", False)
        )

    # Add OpenAI handling
    elif provider == ModelProvider.OPENAI:
        if not config.openai_api_key:
            raise ValueError("openai_api_key must be set for OpenAI provider")
            
        return ChatOpenAI(
            model=model_config["model_name"],
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_new_tokens", 512),
            model_kwargs=model_config.get("model_kwargs", {}),
            openai_api_key=config.openai_api_key
        )

    # Add Anthropic handling 
    elif provider == ModelProvider.ANTHROPIC:
        if not config.anthropic_api_key:
            raise ValueError("anthropic_api_key must be set for Anthropic provider")
            
        return ChatAnthropic(
            model=model_config["model_name"],
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_new_tokens", 512),
            anthropic_api_key=config.anthropic_api_key
        )

    # Add AWS Bedrock handling
    elif provider == ModelProvider.BEDROCK:
        if not (config.aws_access_key_id and config.aws_secret_access_key and config.aws_region):
            raise ValueError("AWS credentials and region must be set for Bedrock provider")
            
        # Use ChatBedrockConverse by default since it's recommended for most users
        return ChatBedrockConverse(
            model=model_config["model_name"],  # e.g. "anthropic.claude-3-sonnet-20240229-v1:0"
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_new_tokens", 512),
            model_kwargs=model_config.get("model_kwargs", {}),
            credentials_profile_name=model_config.get("aws_profile"),
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key
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
    if gate_name not in config["quality_gates"]:
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
