storybook.

A LLM Based Novel Generation Workflow with Consumer Behavioural and Market Trend Analysis Tools - Tasked to Ensure Cohesive Best-Seller Grade Work based on Inference of Consumer Habits, Market Research, Trend Analysis with continuous feedback and review loops measuring standards of Generative Works.

Basic Usage from CLI:
python main.py

Tailored Use from Config:
python main.py --config my_config.json --output ./novel.doc

Resuming Previous Work:
python main.py --input ./output/my-novel-draft.json



Generate Work from Scratch or from Configuration Files (JSON/YAML):

Below is a comprehensive explanation of the configuration file used in the novel generation system, along with the complete implementation.
Configuration File Format
The configuration file is implemented as a Python class using Pydantic for validation, but it can also be loaded from JSON or YAML formats. It defines all customizable parameters for the novel generation process.
Configuration Parameters
LLM Configuration

llm_model: The LLM model to use (default: "gpt-4")
temperature: Temperature for agent LLM calls (default: 0.7)
openai_api_key: OpenAI API key (defaults to environment variable)

Project Settings

project_name: Name of the novel project (default: "New Novel Project")
genre: Primary genre of the novel
target_word_count: Target word count (default: 80,000)
target_audience: Target audience age range (default: "Adult")

Quality Thresholds

min_thematic_coherence: Minimum thematic coherence score (default: 0.75)
min_character_consistency: Minimum character consistency score (default: 0.8)
min_narrative_engagement: Minimum narrative engagement score (default: 0.7)

System Settings

max_revision_cycles: Maximum number of revision cycles (default: 3)
parallel_chapter_generation: Whether to generate chapters in parallel (default: True)

Advanced Agent Settings (Optional)

agent_concurrency: Maximum number of concurrent agent operations
agent_timeout: Timeout for agent operations in seconds
enable_supervision: Enable human supervision for critical decision points

Output Settings (Optional)

output_format: Format for exporting the novel (e.g., "markdown", "docx", "pdf")
chapter_naming_convention: Convention for chapter titles
include_front_matter: Whether to include front matter in exports

Complete Implementation (config.py)
pythonCopy# config.py
import os
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator
import json
import yaml

class Config(BaseModel):
    """Configuration for the novel generation system."""
    # LLM Configuration
    llm_model: str = Field(default="gpt-4", description="The LLM model to use")
    temperature: float = Field(default=0.7, description="Temperature for agent LLM calls")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Project Settings
    project_name: str = Field(default="New Novel Project", description="Name of the novel project")
    genre: str = Field(default="", description="Primary genre of the novel")
    subgenres: List[str] = Field(default_factory=list, description="Secondary genres or subgenres")
    target_word_count: int = Field(default=80000, description="Target word count")
    target_audience: str = Field(default="Adult", description="Target audience age range")
    
    # Quality Thresholds
    min_thematic_coherence: float = Field(default=0.75, description="Minimum thematic coherence score")
    min_character_consistency: float = Field(default=0.8, description="Minimum character consistency score")
    min_narrative_engagement: float = Field(default=0.7, description="Minimum narrative engagement score")
    
    # System Settings
    max_revision_cycles: int = Field(default=3, description="Maximum number of revision cycles")
    parallel_chapter_generation: bool = Field(default=True, description="Whether to generate chapters in parallel")
    
    # Advanced Agent Settings
    agent_concurrency: Optional[int] = Field(default=None, description="Maximum number of concurrent agent operations")
    agent_timeout: Optional[int] = Field(default=None, description="Timeout for agent operations in seconds")
    enable_supervision: bool = Field(default=False, description="Enable human supervision for critical decision points")
    
    # Output Settings
    output_format: Literal["markdown", "docx", "pdf", "txt"] = Field(default="markdown", description="Format for exporting the novel")
    chapter_naming_convention: Literal["numeric", "text", "mixed"] = Field(default="numeric", description="Convention for chapter titles")
    include_front_matter: bool = Field(default=True, description="Whether to include front matter in exports")
    
    # Writing Style Preferences
    narrative_pov: Literal["first_person", "third_person_limited", "third_person_omniscient"] = Field(
        default="third_person_limited", 
        description="Preferred narrative point of view"
    )
    narrative_tense: Literal["past", "present"] = Field(
        default="past", 
        description="Preferred narrative tense"
    )
    prose_style: str = Field(
        default="balanced", 
        description="Preferred prose style (e.g., 'literary', 'conversational', 'balanced', 'minimalist')"
    )
    dialogue_density: Literal["sparse", "moderate", "heavy"] = Field(
        default="moderate", 
        description="Preferred dialogue frequency"
    )
    description_depth: Literal["minimal", "moderate", "detailed"] = Field(
        default="moderate", 
        description="Preferred level of descriptive detail"
    )
    
    # Performance Optimization
    enable_caching: bool = Field(default=True, description="Enable caching of LLM responses")
    cache_directory: str = Field(default=".cache", description="Directory for caching")
    memory_optimization: Literal["low", "medium", "high"] = Field(
        default="medium", 
        description="Level of memory optimization (affects token usage)"
    )
    
    @validator("openai_api_key", pre=True, always=True)
    def validate_openai_api_key(cls, v):
        """Use environment variable if API key not provided."""
        if v is None:
            return os.environ.get("OPENAI_API_KEY")
        return v
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for initializing an LLM."""
        return {
            "model": self.llm_model,
            "temperature": self.temperature,
            "api_key": self.openai_api_key,
        }
    
    def get_style_guide(self) -> Dict[str, Any]:
        """Get the style guide based on configuration settings."""
        return {
            "pov": self.narrative_pov.replace("_", " "),
            "tense": self.narrative_tense,
            "prose_style": self.prose_style,
            "dialogue_density": self.dialogue_density,
            "description_depth": self.description_depth
        }
    
    @classmethod
    def from_file(cls, file_path: str) -> "storybookConfig":
        """Load configuration from a file (JSON or YAML)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config_data = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("Configuration file must be JSON or YAML format")
        
        return cls(**config_data)
    
    def to_file(self, file_path: str) -> None:
        """Save configuration to a file (JSON or YAML)."""
        config_data = self.model_dump()
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.json'):
                json.dump(config_data, f, indent=2)
            elif file_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_data, f, sort_keys=False)
            else:
                raise ValueError("Output file must have .json, .yaml, or .yml extension")

# Default configuration
default_config = storybookConfig()
Sample Configuration Files
Basic JSON Configuration (config.json)
jsonCopy{
  "llm_model": "gpt-4",
  "temperature": 0.7,
  "project_name": "The Lost Kingdom",
  "genre": "Fantasy",
  "subgenres": ["Epic Fantasy", "Adventure"],
  "target_word_count": 95000,
  "target_audience": "Young Adult",
  "max_revision_cycles": 2,
  "narrative_pov": "third_person_limited",
  "narrative_tense": "past",
  "prose_style": "literary",
  "dialogue_density": "moderate",
  "description_depth": "detailed"
}
Advanced YAML Configuration (config.yaml)
yamlCopy# LLM Configuration
llm_model: gpt-4
temperature: 0.7

# Project Settings
project_name: "Corporate Shadows"
genre: "Thriller"
subgenres: 
  - "Psychological Thriller"
  - "Corporate Espionage"
target_word_count: 85000
target_audience: "Adult"

# Quality Thresholds
min_thematic_coherence: 0.8
min_character_consistency: 0.85
min_narrative_engagement: 0.75

# System Settings
max_revision_cycles: 3
parallel_chapter_generation: true

# Advanced Agent Settings
agent_concurrency: 4
agent_timeout: 300
enable_supervision: true

# Output Settings
output_format: "docx"
chapter_naming_convention: "mixed"
include_front_matter: true

# Writing Style Preferences
narrative_pov: "first_person"
narrative_tense: "present"
prose_style: "conversational"
dialogue_density: "heavy"
description_depth: "moderate"

# Performance Optimization
enable_caching: true
cache_directory: ".novel_cache"
memory_optimization: "high"
Usage in Code
To use the configuration in the main application:
pythonCopyimport os
from config import storybookConfig

# Load from environment variables or defaults
config = storybookConfig()

# Or load from a file
config = storybookConfig.from_file("novel_config.json")

# Override specific settings
config.target_word_count = 100000
config.max_revision_cycles = 4

# Save configuration for later use
config.to_file("saved_config.yaml")

# Use in LLM initialization
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(**config.get_llm_kwargs())
Command-Line Integration
The main.py file supports loading a custom configuration file:
bashCopypython main.py --config my_novel_config.json --output ./my_novel
This configuration system provides comprehensive customization of the novel generation process while maintaining sensible defaults for users who prefer not to configure every aspect manually.



A Production of the Adventures of the Persistently Impaired (...and Other Tales).
