from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Dict, Optional, Any, TypedDict
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from enum import Enum

from langchain_core.runnables import RunnableConfig, ensure_config


# Load environment variables from .env file
load_dotenv()


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the storybook system."""

    system_prompt: str = field(
        default="You are part of a specialized novel writing system. Each agent has a specific role in crafting the novel.",
        metadata={
            "description": "The base system prompt for all agents in the storybook system."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="microsoft/phi-2",  # Changed default to Hugging Face model
        metadata={
            "description": "The name of the language model to use for the agents' interactions."
        },
    )

    model_provider: ModelProvider = field(
        default=ModelProvider.HUGGINGFACE,
        metadata={
            "description": "The model provider to use (huggingface or replicate)."
        },
    )

    mongodb_connection_string: str = field(
        default=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"),
        metadata={
            "description": "Connection string for MongoDB checkpointing."
        },
    )

    mongodb_database_name: str = field(
        default=os.getenv("MONGODB_DATABASE_NAME", "storybook_system"),
        metadata={
            "description": "MongoDB database name for checkpointing."
        },
    )

    anthropic_api_key: str = field(
        default=os.getenv("ANTHROPIC_API_KEY", ""),
        metadata={
            "description": "API key for Anthropic (Claude) models."
        },
    )

    openai_api_key: str = field(
        default=os.getenv("OPENAI_API_KEY", ""),
        metadata={
            "description": "API key for OpenAI models."
        },
    )

    google_api_key: str = field(
        default=os.getenv("GOOGLE_API_KEY", ""),
        metadata={
            "description": "API key for Google models."
        },
    )

    huggingface_api_key: str = field(
        default=os.getenv("HUGGINGFACE_API_KEY", ""),
        metadata={
            "description": "API key for Hugging Face models."
        },
    )

    replicate_api_key: str = field(
        default=os.getenv("REPLICATE_API_KEY", ""),
        metadata={
            "description": "API key for Replicate models."
        },
    )

    default_model_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model_name": "microsoft/phi-2",
            "task": "text-generation",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "repetition_penalty": 1.03
        },
        metadata={
            "description": "Default model configuration for all agents."
        },
    )

    agent_model_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "description": "Per-agent model configurations."
        },
    )

    quality_gates: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "initialization_to_development": {
                "required_metrics": ["concept_clarity", "market_viability"],
                "thresholds": {"concept_clarity": 7, "market_viability": 6}
            },
            "development_to_creation": {
                "required_metrics": ["structural_integrity", "character_depth"],
                "thresholds": {"structural_integrity": 7, "character_depth": 7}
            },
            "creation_to_refinement": {
                "required_metrics": ["draft_completion", "narrative_consistency"],
                "thresholds": {"draft_completion": 95, "narrative_consistency": 6}
            },
            "refinement_to_finalization": {
                "required_metrics": ["prose_quality", "plot_coherence"],
                "thresholds": {"prose_quality": 8, "plot_coherence": 7}
            },
            "finalization_to_complete": {
                "required_metrics": ["overall_quality", "market_readiness"],
                "thresholds": {"overall_quality": 8, "market_readiness": 7}
            }
        },
        metadata={
            "description": "Quality thresholds for progressing between phases."
        },
    )

    agent_roles: Dict[str, str] = field(
        default_factory=lambda: {
            "executive_director": "Oversees the entire novel creation process and delegates tasks.",
            "creative_director": "Manages creative aspects including story, characters, and setting.",
            "structure_architect": "Designs the novel's overall structure and pacing.",
            "character_psychology_specialist": "Develops deep, psychologically consistent characters.",
            "human_feedback_manager": "Processes and integrates feedback from human reviewers.",
            "quality_assessment_director": "Evaluates the quality of the novel at various stages.",
            "project_timeline_manager": "Manages the timeline for the novel creation process.",
            "market_alignment_director": "Ensures the novel aligns with current market trends.",
            "plot_development_specialist": "Crafts engaging and coherent plot elements.",
            "world_building_expert": "Creates rich, detailed, and consistent world settings.",
            "character_voice_designer": "Ensures unique and consistent character voices.",
            "character_relationship_mapper": "Designs complex character relationships.",
            "domain_knowledge_specialist": "Provides specialized knowledge in relevant domains.",
            "cultural_authenticity_expert": "Ensures cultural aspects are represented accurately.",
            "content_development_director": "Oversees the development of content elements.",
            "chapter_drafters": "Drafts individual chapters following established structure.",
            "scene_construction_specialists": "Designs and constructs individual scenes.",
            "dialogue_crafters": "Creates engaging and character-appropriate dialogue.",
            "continuity_manager": "Ensures narrative continuity throughout the novel.",
            "voice_consistency_monitor": "Maintains consistent narrative voice and tone.",
            "emotional_arc_designer": "Designs emotional journeys for readers and characters.",
            "editorial_director": "Manages the editorial and revision process.",
            "structural_editor": "Reviews and revises the novel's overall structure.",
            "character_arc_evaluator": "Evaluates the completeness of character arcs.",
            "thematic_coherence_analyst": "Ensures thematic elements are coherent and meaningful.",
            "prose_enhancement_specialist": "Improves prose quality and readability.",
            "dialogue_refinement_expert": "Refines dialogue for authenticity and impact.",
            "rhythm_cadence_optimizer": "Optimizes the rhythm and flow of narrative prose.",
            "grammar_consistency_checker": "Ensures grammatical correctness and consistency.",
            "fact_verification_specialist": "Verifies factual claims in the novel.",
            "positioning_specialist": "Positions the novel effectively for the market.",
            "title_blurb_optimizer": "Optimizes title and marketing blurbs.",
            "differentiation_strategist": "Identifies unique selling points of the novel.",
            "formatting_standards_expert": "Ensures the novel meets formatting standards."
        },
        metadata={
            "description": "Descriptions of each agent's role in the novel creation process."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> 'Configuration':
        """Create configuration from runtime config."""
        if not config:
            return cls()
            
        configurable = config.get("configurable", {})
        
        # Get HF token from runtime config or environment
        hf_token = configurable.get("huggingface_token") or os.getenv("HUGGINGFACE_API_KEY", "")
        
        # Get default model config
        default_model = configurable.get("default_model", cls.default_model_config)
        
        # Get per-agent configurations
        agent_models = configurable.get("agent_models", {})
        
        return cls(
            huggingface_api_key=hf_token,
            default_model_config=default_model,
            agent_model_configs=agent_models,
            # ...other fields...
        )


class ModelProvider(str, Enum):
    """Supported model providers."""
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"  # Add Azure OpenAI

class AgentModelConfig(TypedDict, total=False):
    """Configuration for an individual agent's model."""
    provider: ModelProvider
    model_name: str  # e.g. "microsoft/phi-2" or "replicate/model-id"
    task: str  # e.g. "text-generation"
    max_new_tokens: int 
    temperature: float
    repetition_penalty: float

class StoryBookConfig(TypedDict, total=False):
    """Runtime configuration schema for the storybook system."""
    project_id: str  # Required project identifier
    system_prompt: Optional[str]  # Optional custom system prompt
    model_provider: ModelProvider  # The model provider to use
    huggingface_token: Optional[str]  # HuggingFace API token
    replicate_token: Optional[str]  # Replicate API token
    ollama_base_url: Optional[str]  # Add Ollama base URL config
    llama_cpp_model_path: Optional[str]  # Add path to llama.cpp model weights
    openai_api_key: Optional[str]  # Add OpenAI API key
    anthropic_api_key: Optional[str]  # Add Anthropic API key
    aws_access_key_id: Optional[str]  # Add AWS credentials
    aws_secret_access_key: Optional[str]
    aws_region: Optional[str]
    azure_openai_api_key: Optional[str]  # Add Azure OpenAI credentials
    azure_openai_endpoint: Optional[str]
    azure_deployment_name: Optional[str]
    azure_api_version: Optional[str]
    default_model: Optional[AgentModelConfig]  # Default model config
    agent_models: Optional[Dict[str, AgentModelConfig]]  # Per-agent configs

# Update the graph builder to use this schema
builder = StateGraph(NovelSystemState, config_schema=StoryBookConfig)
