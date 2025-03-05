from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from datetime import datetime
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.llms import Replicate
from storybook.configuration import ModelProvider, Configuration


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
