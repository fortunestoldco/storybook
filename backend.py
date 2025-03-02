from enum import Enum
from typing import Dict, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class BackendProvider(str, Enum):
    """Enum defining the available backend providers for the LLM models."""
    AWS_BEDROCK = "AWS Bedrock"
    HUGGINGFACE = "HuggingFace Endpoint"
    AZURE_OPENAI = "Azure OpenAI"
    GOOGLE_VERTEX = "Google Vertex AI Model Garden"
    OLLAMA = "Ollama"
    LLAMACPP = "Llama.cpp"
    REPLICATE = "Replicate"
    CUSTOM = "Custom"


class BackendConfig(BaseModel):
    """Configuration for LLM backend provider."""
    provider: BackendProvider
    api_key: Optional[str] = None
    api_url: Optional[HttpUrl] = None
    region: Optional[str] = None
    project_id: Optional[str] = None 
    deployment_name: Optional[str] = None
    model_endpoints: Optional[Dict[str, str]] = None


# Default configuration based on environment variables
def get_default_backend_config() -> BackendConfig:
    """Get default backend configuration from environment variables."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    provider_str = os.getenv("BACKEND_PROVIDER", "AWS Bedrock")
    
    try:
        provider = BackendProvider(provider_str)
    except ValueError:
        # Default to AWS Bedrock if invalid provider
        provider = BackendProvider.AWS_BEDROCK
    
    config = BackendConfig(
        provider=provider,
        api_key=os.getenv("BACKEND_API_KEY"),
        api_url=os.getenv("BACKEND_API_URL"),
        region=os.getenv("AWS_REGION") or os.getenv("BACKEND_REGION"),
        project_id=os.getenv("GOOGLE_PROJECT_ID") or os.getenv("BACKEND_PROJECT_ID"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME") or os.getenv("BACKEND_DEPLOYMENT_NAME"),
        model_endpoints={}
    )
    
    # Load any model-specific endpoints if provided
    # Format in env: MODEL_ENDPOINT_claude-3-opus=https://...
    for key, value in os.environ.items():
        if key.startswith("MODEL_ENDPOINT_"):
            model_name = key.replace("MODEL_ENDPOINT_", "")
            config.model_endpoints[model_name] = value
    
    return config
