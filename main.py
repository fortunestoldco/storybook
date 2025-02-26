# storybook/main.py

"""
Main entry point for the Storybook novel generation system.
"""

import argparse
import json
import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer

from storybook.config.system_config import SystemConfig, DEFAULT_CONFIG
from storybook.config.models_config import ModelsConfig, DEFAULT_MODELS_CONFIG
from storybook.agents.base_agent import BaseAgent
from storybook.agents.project_management import ProjectLeadAgent, MarketResearchAgent
from storybook.agents.story_architecture import StructureSpecialistAgent, PlotDevelopmentAgent
from storybook.agents.character_development import CharacterPsychologyAgent, CharacterRelationshipMapper
from storybook.agents.writing import ChapterWriterAgent
from storybook.agents.editing import StructuralEditorAgent, ProseEnhancementAgent, ContinuityCheckerAgent
from storybook.tools.nlp.minilm_analyzer import MiniLMAnalyzer
from storybook.tools.nlp.voice_analyzer import VoiceAnalyzer
from storybook.workflows.master_workflow import MasterWorkflow
from storybook.states.project_state import ProjectState
from storybook.innovations.voice.adaptive_voice import AdaptiveVoiceResonance
from storybook.innovations.creativity.emergence import CreativeEmergenceFacilitator

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Load system configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        SystemConfig instance
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return SystemConfig(**config_data)
    return DEFAULT_CONFIG

def load_models_config(config_path: Optional[str] = None) -> ModelsConfig:
    """
    Load models configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ModelsConfig instance
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return ModelsConfig(**config_data)
    return DEFAULT_MODELS_CONFIG

def initialize_language_models(models_config: ModelsConfig) -> Dict[str, Any]:
    """
    Initialize language models based on configuration.
    
    Args:
        models_config: Models configuration
        
    Returns:
        Dictionary of initialized language models
    """
    models = {}
    
    for model_name, model_config in models_config.models.items():
        if model_config.provider == "openai":
            models[model_name] = ChatOpenAI(
                model=model_config.name,
                temperature=model_config.parameters.get("temperature", 0.7),
                max_tokens=model_config.parameters.get("max_tokens", 2000)
            )
        elif model_config.provider == "anthropic":
            models[model_name] = ChatAnthropic(
                model=model_config.name,
                temperature=model_config.parameters.get("temperature", 0.7),
                max_tokens=model_config.parameters.get("max_tokens", 4000)
            )
    
    return models

def initialize_tools(config: SystemConfig) -> Dict[str, Any]:
    """
    Initialize tools based on configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Dictionary of initialized tools
    """
    tools = {}
    
    # Initialize NLP tools
    tools["nlp"] = {}
    
    # Initialize MiniLM analyzer
    minilm_model = config.nlp.minilm_model
    tools["nlp"]["minilm_analyzer"] = MiniLMAnalyzer(model_name=minilm_model)
    
    # Initialize voice analyzer
    embedding_model = SentenceTransformer(minilm_model)
    tools["nlp"]["voice_analyzer"] = VoiceAnalyzer(embedding_model=embedding_model)
    
    # Initialize innovations
    tools["innovations"] = {}
    tools["innovations"]["adaptive_voice"] = AdaptiveVoiceResonance(
        voice_analyzer=tools["nlp"]["voice_analyzer"]
    )
    tools["innovations"]["creative_emergence"] = CreativeEmergenceFacilitator()
    
    return tools

def initialize_agents(
    config: SystemConfig,
    models: Dict[str, Any],
    tools: Dict[str, Any]
) -> Dict[str, BaseAgent]:
    """
    Initialize agents based on configuration.
    
    Args:
        config: System configuration
        models: Dictionary of language models
        tools: Dictionary of tools
        
    Returns:
        Dictionary of initialized agents
    """
    agents = {}
    default_model = models[config.agents.model]
    minilm_analyzer = tools["nlp"]["minilm_analyzer"]
    
    # Initialize project management agents
    agents["project_lead"] = ProjectLeadAgent(
        name="Project Lead",
        description="Leads the novel project and makes high-level decisions",
        system_prompt="You are the Project Lead for a best-selling novel generation system. Your role is to guide the overall direction of the novel and make high-level decisions about genre, target audience, and major themes.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    agents["market_research"] = MarketResearchAgent(
        name="Market Research Specialist",
        description="Researches market trends and reader preferences",
        system_prompt="You are the Market Research Specialist for a best-selling novel generation system. Your role is to analyze current market trends, reader preferences, and successful books in the target genre to inform the novel's creation.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    # Initialize story architecture agents
    agents["structure_specialist"] = StructureSpecialistAgent(
        name="Structure Specialist",
        description="Designs the novel's narrative structure",
        system_prompt="You are the Structure Specialist for a best-selling novel generation system. Your role is to design the overall narrative structure, including acts, key plot points, and chapter outlines.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    agents["plot_development"] = PlotDevelopmentAgent(
        name="Plot Development Specialist",
        description="Develops detailed plot elements and arcs",
        system_prompt="You are the Plot Development Specialist for a best-selling novel generation system. Your role is to develop detailed plot elements, narrative arcs, and ensure engaging story progression.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    # Initialize character development agents
    agents["character_psychology"] = CharacterPsychologyAgent(
        name="Character Psychology Specialist",
        description="Develops psychologically complex characters",
        system_prompt="You are the Character Psychology Specialist for a best-selling novel generation system. Your role is to develop psychologically complex and believable characters with depth, motivations, and internal conflicts.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    agents["character_relationship"] = CharacterRelationshipMapper(
        name="Character Relationship Mapper",
        description="Maps relationships between characters",
        system_prompt="You are the Character Relationship Mapper for a best-selling novel generation system. Your role is to design and track complex, dynamic relationships between characters that drive the plot and character development.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    # Initialize writing agents
    agents["chapter_writer"] = ChapterWriterAgent(
        name="Chapter Writer",
        description="Writes individual chapters of the novel",
        system_prompt="You are the Chapter Writer for a best-selling novel generation system. Your role is to write engaging, high-quality chapters based on the novel's outline and character profiles.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    # Initialize editing agents
    agents["structural_editor"] = StructuralEditorAgent(
        name="Structural Editor",
        description="Evaluates and refines the novel's structure",
        system_prompt="You are the Structural Editor for a best-selling novel generation system. Your role is to evaluate and refine the novel's structure, ensuring proper pacing, tension, and narrative flow.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    agents["prose_enhancement"] = ProseEnhancementAgent(
        name="Prose Enhancement Specialist",
        description="Refines and enhances prose quality",
        system_prompt="You are the Prose Enhancement Specialist for a best-selling novel generation system. Your role is to refine and enhance the prose quality, ensuring engaging, polished writing with a consistent voice.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    agents["continuity_checker"] = ContinuityCheckerAgent(
        name="Continuity Checker",
        description="Ensures narrative continuity",
        system_prompt="You are the Continuity Checker for a best-selling novel generation system. Your role is to ensure narrative continuity, checking for inconsistencies in plot, character details, setting, and timeline.",
        llm=default_model,
        tools=tools,
        config=config.agents,
        minilm_analyzer=minilm_analyzer
    )
    
    return agents

def load_project(project_path: str) -> Dict[str, Any]:
    """
    Load project from file.
    
    Args:
        project_path: Path to project file
        
    Returns:
        Project data
    """
    if os.path.exists(project_path):
        with open(project_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """Main entry point for the system."""
    parser = argparse.ArgumentParser(description="Storybook - Best-Seller Novel Generation System")
    parser.add_argument('--config', default=None, help='Path to system configuration')
    parser.add_argument('--models', default=None, help='Path to models configuration')
    parser.add_argument('--project', default=None, help='Path to project file')
    parser.add_argument('--output', default='output', help='Output directory')
    args = parser.parse_args()
    
    # Load configurations
    system_config = load_config(args.config)
    models_config = load_models_config(args.models)
    
    # Initialize language models
    models = initialize_language_models(models_config)
    
    # Initialize tools
    tools = initialize_tools(system_config)
    
    # Initialize agents
    agents = initialize_agents(system_config, models, tools)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize workflow
    workflow = MasterWorkflow(agents, config=system_config.workflows.dict())
    
    # Load or create project
    project_data = load_project(args.project) if args.project else {}
    
    # Set up initial state
    initial_state = {
        "project_state": project_data,
        "project_parameters": {
            "title": "Untitled Novel",
            "genre": "thriller",
            "target_audience": "adult",
            "word_count_target": 80000
        }
    }
    
    # Run workflow
    result = workflow.run(initial_state)
    
    # Save results
    output_path = os.path.join(args.output, f"{result['project_state']['project_id']}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    # Save manuscript
    project_state = ProjectState(result["project_state"])
    manuscript_path = os.path.join(args.output, f"{result['project_state']['title']}.txt")
    with open(manuscript_path, 'w') as f:
        f.write(project_state.get_full_manuscript())
        
    print(f"Novel generation complete. Output saved to: {output_path}")
    print(f"Manuscript saved to: {manuscript_path}")

if __name__ == "__main__":
    main()
