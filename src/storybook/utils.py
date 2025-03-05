from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from datetime import datetime
import os


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


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name."""
    # Handle case where provider is not specified
    if "/" not in fully_specified_name:
        provider = "openai"  # Default provider
        model = fully_specified_name
    else:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    
    # Set the appropriate API key for the model provider
    api_key = None
    if provider.lower() == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider.lower() == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
    
    return init_chat_model(model, model_provider=provider, api_key=api_key)


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
