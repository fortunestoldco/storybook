# storybook/config/system_config.py

"""
Configuration for the Storybook system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for LLM agents."""
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for generation")
    model: str = Field(default="gpt-4", description="Model to use for this agent")
    streaming: bool = Field(default=False, description="Whether to stream output")
    self_evaluation_enabled: bool = Field(default=True, description="Enable self-evaluation")
    improvement_iterations: int = Field(default=2, description="Number of improvement iterations")

class ToolConfig(BaseModel):
    """Configuration for tools."""
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")

class WorkflowConfig(BaseModel):
    """Configuration for workflows."""
    timeout_seconds: int = Field(default=3600, description="Timeout in seconds")
    max_iterations_per_phase: int = Field(default=5, description="Maximum iterations per phase")
    quality_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "draft": 0.65,
            "revision": 0.75,
            "final": 0.85
        },
        description="Quality thresholds for different stages"
    )

class NLPConfig(BaseModel):
    """Configuration for NLP models."""
    minilm_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="MiniLM model to use"
    )
    emotional_threshold: float = Field(default=0.75, description="Emotional impact threshold")
    voice_consistency_threshold: float = Field(default=0.85, description="Voice consistency threshold")
    thematic_coherence_threshold: float = Field(default=0.8, description="Thematic coherence threshold")

class StorageConfig(BaseModel):
    """Configuration for storage."""
    version_control: bool = Field(default=True, description="Enable version control")
    backup_frequency_minutes: int = Field(default=15, description="Backup frequency in minutes")

class SystemConfig(BaseModel):
    """Main system configuration."""
    name: str = Field(default="Storybook - Best-Seller Novel Generation System")
    version: str = Field(default="1.0.0")
    agents: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")
    tools: Dict[str, ToolConfig] = Field(default_factory=dict, description="Tool configuration")
    workflows: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    nlp: NLPConfig = Field(default_factory=NLPConfig, description="NLP configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage configuration")

# Default system configuration
DEFAULT_CONFIG = SystemConfig()
