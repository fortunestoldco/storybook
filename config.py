import os
from dotenv import load_dotenv

# Load environment variables from .env file or Secret Manager
_ = load_dotenv("../.env")

# Environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "storybook")
LANGSMITH_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING") == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Storybook quality gates configuration
QUALITY_GATES = {
    "initialization": 0.7,
    "development": 0.7,
    "creation": 0.7,
    "refinement": 0.8,
    "finalization": 0.8,
    "initialization_to_development": {
        "planning_quality": 0.7,
        "market_alignment": 0.6
    },
    "development_to_creation": {
        "structure_quality": 0.7,
        "character_depth": 0.7,
        "world_building": 0.7
    },
    "creation_to_refinement": {
        "content_quality": 0.7,
        "narrative_flow": 0.7,
        "dialogue_quality": 0.7
    },
    "refinement_to_finalization": {
        "editing_quality": 0.8,
        "prose_quality": 0.8,
        "thematic_coherence": 0.7
    },
    "finalization_to_complete": {
        "market_readiness": 0.8,
        "overall_quality": 0.8
    }
}

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "model_id": "HuggingFaceH4/zephyr-7b-beta",
    "task": "text-generation",
    "temperature": 0.1,
    "max_new_tokens": 512,
    "do_sample": True,
    "repetition_penalty": 1.03
}

# Model choices for the UI
MODEL_CHOICES = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-chat-hf",
    "google/gemma-7b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "facebook/bart-large-cnn",
    "gpt2",
    "distilgpt2",
    "microsoft/phi-2"
]

def get_storybook_config():
    """Return the full storybook configuration dictionary"""
    return {
        "MONGODB_URI": MONGODB_URI,
        "mongodb_database_name": "storybook",
        "quality_gates": QUALITY_GATES
    }

def create_agent_model_configs():
    """Create model configurations for each agent based on optimal performance characteristics."""
    return {
        # Directors
        "executive_director": {
            "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "creative_director": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "editorial_director": {
            "model_id": "meta-llama/Llama-2-13b-chat-hf",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "content_development_director": {
            "model_id": "google/gemma-7b-it",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "market_alignment_director": {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },

        # Creative Team
        "structure_architect": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "plot_development_specialist": {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "world_building_expert": {
            "model_id": "google/gemma-7b-it",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "character_psychology_specialist": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "character_voice_designer": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "character_relationship_mapper": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },

        # Content Team
        "chapter_drafters": {
            "model_id": "meta-llama/Llama-2-13b-chat-hf",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "scene_construction_specialists": {
            "model_id": "google/gemma-7b-it",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "dialogue_crafters": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "continuity_manager": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "voice_consistency_monitor": {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "emotional_arc_designer": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },

        # Editorial Team
        "prose_enhancement_specialist": {
            "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "dialogue_refinement_expert": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "structural_editor": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "character_arc_evaluator": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "thematic_coherence_analyst": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "rhythm_cadence_optimizer": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "grammar_consistency_checker": {
            "model_id": "google/flan-t5-large",
            "task": "text2text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "fact_verification_specialist": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },

        # Research Team
        "domain_knowledge_specialist": {
            "model_id": "google/flan-t5-xl",
            "task": "text2text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "cultural_authenticity_expert": {
            "model_id": "google/flan-t5-xl",
            "task": "text2text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },

        # Marketing Team
        "positioning_specialist": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "title_blurb_optimizer": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "differentiation_strategist": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        },
        "formatting_standards_expert": {
            "model_id": "microsoft/phi-2",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.03
        }
    }

# Agent team groupings for UI
AGENT_TEAMS = {
    "Directors": ["executive_director", "creative_director", "editorial_director",
                 "content_development_director", "market_alignment_director"],
    "Creative Team": ["structure_architect", "plot_development_specialist", "world_building_expert",
                    "character_psychology_specialist", "character_voice_designer", "character_relationship_mapper"],
    "Content Team": ["chapter_drafters", "scene_construction_specialists", "dialogue_crafters",
                   "continuity_manager", "voice_consistency_monitor", "emotional_arc_designer"],
    "Editorial Team": ["structural_editor", "character_arc_evaluator", "thematic_coherence_analyst",
                     "prose_enhancement_specialist", "dialogue_refinement_expert", "rhythm_cadence_optimizer",
                     "grammar_consistency_checker", "fact_verification_specialist"],
    "Research Team": ["domain_knowledge_specialist", "cultural_authenticity_expert"],
    "Marketing Team": ["positioning_specialist", "title_blurb_optimizer", "differentiation_strategist",
                     "formatting_standards_expert"]
}

# All available agents list
ALL_AGENTS = [agent for team_agents in AGENT_TEAMS.values() for agent in team_agents]