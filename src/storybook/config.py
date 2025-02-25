from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class ProjectConfig(BaseModel):
    """Configuration for the novel writing project."""
    novel_id: str
    title: str
    word_count: int
    chapter_count: int
    guidelines: List[str]
    notes: Optional[str]
    genre: Optional[str]
    target_audience: Optional[str]
    deadline: Optional[datetime]


class MongoDBConfig(BaseModel):
    """Configuration for MongoDB Atlas."""
    connection_string: str
    database_name: str = "storybook"
    collections: Dict[str, str] = {
        "vectors": "document_vectors",
        "projects": "projects",
        "story_bible": "story_bible",
        "drafts": "drafts",
        "feedback": "feedback",
        "character": "characters",
        "world_building": "world_building"
    }
    vector_search: Dict[str, Any] = {
        "index": "default",
        "num_candidates": 100,
        "embedding_dimension": 1536  # For OpenAI embeddings
    }

class ToolConfig(BaseModel):
    """Configuration for tools used by agents."""
    name: str
    description: str
    parameters: Dict[str, Any]
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None

class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    role: str
    description: str
    tools: List[str]
    temperature: float = 0.7
    max_iterations: int = 3
    memory_type: str = "chat_memory"
    requires_supervision: bool = True
    supervisor: Optional[str] = None

class TeamConfig(BaseModel):
    """Configuration for agent teams."""
    name: str
    supervisor: str
    members: List[str]
    workflow: List[str]
    parallel_execution: bool = False

# Base project directory
BASE_DIR = Path(__file__).parent.absolute()

# MongoDB Atlas configuration
MONGODB_CONFIG = MongoDBConfig(
    connection_string="mongodb+srv://<username>:<password>@cluster.mongodb.net/",
    vector_search={
        "index": "vector_index",
        "num_candidates": 100,
        "embedding_dimension": 1536,
        "similarity_metric": "cosine"  # or "euclidean" or "dotProduct"
    }
)

# Tool configurations
TOOL_CONFIGS = {
    "project_management": ToolConfig(
        name="project_management",
        description="Manages project status, timelines, and coordination",
        parameters={"status_update_interval": 300}
    ),
    "document_retrieval": ToolConfig(
        name="document_retrieval",
        description="Retrieves documents from vector store",
        parameters={"max_results": 5},
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY"
    ),
    "web_crawler": ToolConfig(
        name="web_crawler",
        description="Crawls web for research information",
        parameters={"max_depth": 3, "timeout": 30}
    ),
    "vector_store": ToolConfig(
        name="vector_store",
        description="Stores and retrieves vector embeddings",
        parameters={"collection_name": "embeddings"},
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY"
    ),
    "chat": ToolConfig(
        name="chat",
        description="Handles interactive chat sessions",
        parameters={"max_history": 50}
    )
}

# Agent configurations
AGENT_CONFIGS = {
    "overall_supervisor": AgentConfig(
        name="Overall Supervisor",
        role="project_manager",
        description="Coordinates all teams and manages the overall novel writing process",
        tools=["project_management", "status_updates", "document_management"],
        requires_supervision=False
    ),
    "author_relations": AgentConfig(
        name="Author Relations Agent",
        role="author_liaison",
        description="Handles communication with the author and manages brainstorming sessions",
        tools=["chat", "document_retrieval", "vector_store"],
        supervisor="overall_supervisor"
    ),
    "research_supervisor": AgentConfig(
        name="Research Team Supervisor",
        role="research_manager",
        description="Coordinates research activities and manages research team",
        tools=["project_management", "document_retrieval"],
        supervisor="overall_supervisor"
    ),
    "contextual_researcher": AgentConfig(
        name="Contextual Research Agent",
        role="researcher",
        description="Conducts contextual research for the novel",
        tools=["document_retrieval", "web_crawler"],
        supervisor="research_supervisor"
    ),
    "market_researcher": AgentConfig(
        name="Market Research Agent",
        role="researcher",
        description="Analyzes market trends and competition",
        tools=["document_retrieval", "web_crawler"],
        supervisor="research_supervisor"
    ),
    "consumer_insights": AgentConfig(
        name="Consumer Insights Agent",
        role="analyst",
        description="Analyzes consumer preferences and trends",
        tools=["document_retrieval", "vector_store"],
        supervisor="research_supervisor"
    ),
    "writing_supervisor": AgentConfig(
        name="Writing Team Supervisor",
        role="writing_manager",
        description="Coordinates writing activities and manages writing team",
        tools=["project_management", "document_management"],
        supervisor="overall_supervisor"
    ),
    "world_builder": AgentConfig(
        name="World Builder Agent",
        role="world_designer",
        description="Creates and maintains the novel's world specifications",
        tools=["document_retrieval", "vector_store"],
        supervisor="writing_supervisor"
    ),
    "character_builder": AgentConfig(
        name="Character Builder Agent",
        role="character_designer",
        description="Creates and maintains character specifications",
        tools=["document_retrieval", "vector_store"],
        supervisor="writing_supervisor"
    ),
    "story_writer": AgentConfig(
        name="Story Writer Agent",
        role="writer",
        description="Writes the main narrative content",
        tools=["document_retrieval"],
        supervisor="writing_supervisor"
    ),
    "dialogue_writer": AgentConfig(
        name="Dialogue Writer Agent",
        role="writer",
        description="Writes and refines character dialogue",
        tools=["document_retrieval"],
        supervisor="writing_supervisor"
    ),
    "publishing_supervisor": AgentConfig(
        name="Publishing Team Supervisor",
        role="publishing_manager",
        description="Coordinates publishing activities and manages publishing team",
        tools=["project_management", "document_management"],
        supervisor="overall_supervisor"
    ),
    "consistency_checker": AgentConfig(
        name="Consistency Checker Agent",
        role="editor",
        description="Checks for consistency in plot, characters, and world-building",
        tools=["document_retrieval"],
        supervisor="publishing_supervisor"
    ),
    "continuity_checker": AgentConfig(
        name="Continuity Checker Agent",
        role="editor",
        description="Checks for continuity between chapters and scenes",
        tools=["document_retrieval"],
        supervisor="publishing_supervisor"
    ),
    "editor": AgentConfig(
        name="Editor Agent",
        role="editor",
        description="Performs comprehensive editorial review",
        tools=["document_retrieval"],
        supervisor="publishing_supervisor"
    ),
    "finalisation": AgentConfig(
        name="Finalisation Agent",
        role="publisher",
        description="Prepares manuscript for publication",
        tools=["document_management"],
        supervisor="publishing_supervisor"
    )
}

# Team configurations
TEAM_CONFIGS = {
    "supervisor_team": TeamConfig(
        name="Supervisor Team",
        supervisor="overall_supervisor",
        members=["overall_supervisor"],
        workflow=["initialize_project", "coordinate_teams", "monitor_progress"]
    ),
    "author_relations_team": TeamConfig(
        name="Author Relations Team",
        supervisor="overall_supervisor",
        members=["author_relations"],
        workflow=["brainstorm", "gather_requirements", "collect_feedback"]
    ),
    "research_team": TeamConfig(
        name="Research Team",
        supervisor="research_supervisor",
        members=["contextual_researcher", "market_researcher", "consumer_insights"],
        workflow=["research_context", "analyze_market", "generate_insights"],
        parallel_execution=True
    ),
    "writing_team": TeamConfig(
        name="Writing Team",
        supervisor="writing_supervisor",
        members=["world_builder", "character_builder", "story_writer", "dialogue_writer"],
        workflow=["world_building", "character_development", "writing_chapters"]
    ),
    "publishing_team": TeamConfig(
        name="Publishing Team",
        supervisor="publishing_supervisor",
        members=["consistency_checker", "continuity_checker", "editor", "finalisation"],
        workflow=["check_consistency", "check_continuity", "edit", "finalize"]
    )
}

# Project status states
PROJECT_STATES = [
    "BRAINSTORMING",
    "RESEARCH",
    "WORLD_BUILDING",
    "CHARACTER_DEVELOPMENT",
    "FIRST_DRAFT",
    "SECOND_DRAFT",
    "THIRD_DRAFT",
    "EDITING",
    "FINAL_REVIEW",
    "COMPLETED"
]

# Environment variables required
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "MONGODB_USERNAME",
    "MONGODB_PASSWORD",
    "MONGODB_CLUSTER",
]
