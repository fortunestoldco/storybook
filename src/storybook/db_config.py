import os
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
from .configuration import Configuration

# Load environment variables
load_dotenv()

# Singleton client and configuration
_client: Optional[MongoClient] = None
_config: Optional[Configuration] = None

def initialize_config(config: Configuration):
    """Initialize database configuration."""
    global _config
    _config = config

def get_client() -> MongoClient:
    """Get or create MongoDB client singleton."""
    global _client, _config
    if (_client is None) and (_config is not None):
        _client = MongoClient(_config.mongodb_connection_string)
    return _client

def get_database() -> Database:
    """Get the storybook database."""
    global _config
    if _config is None:
        raise ValueError("Database configuration not initialized. Call initialize_config first.")
    client = get_client()
    return client[_config.mongodb_database_name]

def get_collection(collection_name: str) -> Collection:
    """Get a specific collection from the database."""
    db = get_database()
    return db[collection_name]

def close_connection():
    """Close the MongoDB connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None

# Collection names
COLLECTIONS = {
    "projects": "projects",
    "timelines": "timelines",
    "characters": "characters",
    "character_relationships": "character_relationships",
    "world_building": "world_building", 
    "plot_elements": "plot_elements",
    "story_structures": "story_structures",
    "scenes": "scenes",
    "chapters": "chapters",
    "feedback": "feedback",
    "revisions": "revisions",
    "quality_metrics": "quality_metrics",
    "communications": "communications",
    "market_research": "market_research",
    "style_guides": "style_guides",
    "chat_history": "chat_history",
    "continuity_facts": "continuity_facts",
    "research_reports": "research_reports",
    "research_iterations": "research_iterations",
}

# Vector Search Collections
VECTOR_COLLECTIONS = {
    "style_references": "style_references_vector",
    "theme_elements": "theme_elements_vector",
    "character_voices": "character_voices_vector",
    "market_positioning": "market_positioning_vector",
}

# Create indexes for the collections
def setup_indexes():
    """Setup indexes for all collections."""
    # Projects collection indexes
    projects_col = get_collection(COLLECTIONS["projects"])
    projects_col.create_index("project_id", unique=True)
    
    # Characters collection indexes
    characters_col = get_collection(COLLECTIONS["characters"])
    characters_col.create_index("character_id", unique=True)
    characters_col.create_index("project_id")
    
    # Character relationships collection indexes
    relationships_col = get_collection(COLLECTIONS["character_relationships"])
    relationships_col.create_index([("character1_id", 1), ("character2_id", 1)])
    relationships_col.create_index("project_id")
    
    # Scenes collection indexes
    scenes_col = get_collection(COLLECTIONS["scenes"])
    scenes_col.create_index("scene_id", unique=True)
    scenes_col.create_index("chapter_id")
    scenes_col.create_index("project_id")
    
    # Chapters collection indexes
    chapters_col = get_collection(COLLECTIONS["chapters"])
    chapters_col.create_index("chapter_id", unique=True)
    chapters_col.create_index("project_id")
    
    # World building collection indexes
    world_col = get_collection(COLLECTIONS["world_building"])
    world_col.create_index("entry_id", unique=True)
    world_col.create_index("project_id")
    world_col.create_index("category")
    
    # Continuity facts collection indexes
    continuity_col = get_collection(COLLECTIONS["continuity_facts"])
    continuity_col.create_index("fact_id", unique=True)
    continuity_col.create_index("project_id")
    continuity_col.create_index("entity_id")
    
    # Chat history collection indexes
    chat_col = get_collection(COLLECTIONS["chat_history"])
    chat_col.create_index("session_id")
    
    # Feedback collection indexes
    feedback_col = get_collection(COLLECTIONS["feedback"])
    feedback_col.create_index("feedback_id", unique=True)
    feedback_col.create_index("project_id")
    feedback_col.create_index("category")
    
    print("Database indexes created successfully")
