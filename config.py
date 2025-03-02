# Model configurations for agents
MODEL_CONFIGS = {
    # Strategic Level
    "executive_director": {
        "model": "anthropic/claude-3-opus",
        "temperature": 0.2,
        "max_tokens": 4000,
    },
    "human_feedback_manager": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
    "quality_assessment_director": {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "temperature": 0.2,
        "max_tokens": 3000,
    },
    "project_timeline_manager": {
        "model": "anthropic/claude-3-haiku",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
    
    # Creative Director and Teams
    "creative_director": {
        "model": "anthropic/claude-3-opus",
        "temperature": 0.4,
        "max_tokens": 4000,
    },
    "structure_architect": {
        "model": "databricks/dbrx-instruct",
        "temperature": 0.3,
        "max_tokens": 3000,
    },
    "plot_development_specialist": {
        "model": "google/gemma-7b-it",
        "temperature": 0.5,
        "max_tokens": 2500,
    },
    "world_building_expert": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.6,
        "max_tokens": 3000,
    },
    "character_psychology_specialist": {
        "model": "Qwen/Qwen1.5-72B-Chat",
        "temperature": 0.4,
        "max_tokens": 3500,
    },
    "character_voice_designer": {
        "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "temperature": 0.5,
        "max_tokens": 2500,
    },
    "character_relationship_mapper": {
        "model": "mistralai/mistral-7b-instruct-v0.2",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "emotional_arc_designer": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.5,
        "max_tokens": 2500,
    },
    "reader_attachment_specialist": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "scene_emotion_calibrator": {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "temperature": 0.4,
        "max_tokens": 2000,
    },
    
    # Content Development Director and Teams
    "content_development_director": {
        "model": "mistralai/mixtral-8x7b-instruct-v0.1",
        "temperature": 0.3,
        "max_tokens": 3500,
    },
    "domain_knowledge_specialist": {
        "model": "anthropic/claude-3-opus",
        "temperature": 0.2,
        "max_tokens": 4000,
    },
    "cultural_authenticity_expert": {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "temperature": 0.3,
        "max_tokens": 3500,
    },
    "historical_context_researcher": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.2,
        "max_tokens": 3000,
    },
    "chapter_drafters": {
        "model": "Salesforce/xgen-7b-8k-inst",
        "temperature": 0.6,
        "max_tokens": 6000,
    },
    "scene_construction_specialists": {
        "model": "google/gemma-7b-it",
        "temperature": 0.5,
        "max_tokens": 3000,
    },
    "dialogue_crafters": {
        "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "temperature": 0.6,
        "max_tokens": 3000,
    },
    "continuity_manager": {
        "model": "mistralai/mistral-7b-instruct-v0.2",
        "temperature": 0.3,
        "max_tokens": 2500,
    },
    "voice_consistency_monitor": {
        "model": "anthropic/claude-3-haiku",
        "temperature": 0.3,
        "max_tokens": 2000,
    },
    "description_enhancement_specialist": {
        "model": "microsoft/Phi-3-medium-4k-instruct",
        "temperature": 0.5,
        "max_tokens": 2500,
    },
    
    # Editorial Director and Teams
    "editorial_director": {
        "model": "anthropic/claude-3-opus",
        "temperature": 0.2,
        "max_tokens": 4000,
    },
    "structural_editor": {
        "model": "databricks/dbrx-instruct",
        "temperature": 0.3,
        "max_tokens": 3000,
    },
    "character_arc_evaluator": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.3,
        "max_tokens": 2500,
    },
    "thematic_coherence_analyst": {
        "model": "mistralai/mistral-7b-instruct-v0.2",
        "temperature": 0.3,
        "max_tokens": 2500,
    },
    "prose_enhancement_specialist": {
        "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "dialogue_refinement_expert": {
        "model": "google/gemma-7b-it",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "rhythm_cadence_optimizer": {
        "model": "microsoft/Phi-3-medium-4k-instruct",
        "temperature": 0.4,
        "max_tokens": 2000,
    },
    "grammar_consistency_checker": {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "fact_verification_specialist": {
        "model": "anthropic/claude-3-opus",
        "temperature": 0.1,
        "max_tokens": 3000,
    },
    "formatting_standards_expert": {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    
    # Market Alignment Director and Teams
    "market_alignment_director": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.3,
        "max_tokens": 3000,
    },
    "zeitgeist_analyst": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.4,
        "max_tokens": 3000,
    },
    "cultural_conversation_mapper": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "trend_forecaster": {
        "model": "mistralai/mixtral-8x7b-instruct-v0.1",
        "temperature": 0.4,
        "max_tokens": 3000,
    },
    "hook_optimization_expert": {
        "model": "google/gemma-7b-it",
        "temperature": 0.5,
        "max_tokens": 2500,
    },
    "page_turner_designer": {
        "model": "anthropic/claude-3-haiku",
        "temperature": 0.4,
        "max_tokens": 2000,
    },
    "satisfaction_engineer": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.4,
        "max_tokens": 2500,
    },
    "positioning_specialist": {
        "model": "anthropic/claude-3-sonnet",
        "temperature": 0.3,
        "max_tokens": 2500,
    },
    "title_blurb_optimizer": {
        "model": "google/gemma-7b-it",
        "temperature": 0.5,
        "max_tokens": 2000,
    },
    "differentiation_strategist": {
        "model": "databricks/dbrx-instruct",
        "temperature": 0.3,
        "max_tokens": 2500,
    },
}

# MongoDB configuration
MONGODB_CONFIG = {
    "connection_string": "mongodb://mongodb:27017/",  # Updated for Docker deployment
    "database_name": "novel_writing_system",
    "collections": {
        "project_state": "project_state",
        "documents": "documents",
        "research": "research",
        "feedback": "feedback",
        "metrics": "metrics",
    }
}

# Server configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
}

# Phase thresholds and quality gates
QUALITY_GATES = {
    "initialization_to_development": {
        "project_setup_completion": 100,  # Percentage
        "initial_research_depth": 70,     # Percentage
        "human_approval_required": True,
    },
    "development_to_creation": {
        "character_development_completion": 90,  # Percentage
        "structure_planning_completion": 85,     # Percentage
        "world_building_completion": 80,         # Percentage
        "human_approval_required": True,
    },
    "creation_to_refinement": {
        "draft_completion": 100,            # Percentage
        "narrative_coherence_score": 75,    # Out of 100
        "character_consistency_score": 80,  # Out of 100
        "human_approval_required": True,
    },
    "refinement_to_finalization": {
        "developmental_editing_completion": 100,  # Percentage
        "line_editing_completion": 100,          # Percentage
        "technical_editing_completion": 100,     # Percentage
        "overall_quality_score": 85,             # Out of 100
        "human_approval_required": True,
    },
    "finalization_to_complete": {
        "marketing_package_completion": 100,  # Percentage
        "final_quality_score": 90,            # Out of 100
        "human_final_approval": True,
    },
}

# Import comprehensive prompts from prompts.py
try:
    from prompts import AGENT_PROMPTS as PROMPT_TEMPLATES
except ImportError:
    # Fallback basic prompts if prompts.py is not available
    PROMPT_TEMPLATES = {
        "executive_director": (
            "You are the Executive Director Agent, the system controller for a novel writing system.\n\n"
            "Current Project State:\n"
            "{project_state}\n\n"
            "Current Phase: {current_phase}\n\n"
            "Task: {task}\n\n"
            "Based on the current state and task, provide strategic direction for the project."
        ),
        # Additional prompt templates would be defined here for other agents
    }