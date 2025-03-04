#!/bin/bash

# Create necessary directories
mkdir -p storybook

# Save files
cat > storybook/state.py << 'EOF'
"""Define the state structures for the storybook system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class ProjectInfo:
    """Information about the novel project."""
    
    title: str = ""
    genre: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    length_target: Dict[str, Any] = field(default_factory=dict)
    content_guidelines: Dict[str, Any] = field(default_factory=dict)
    timeline: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputState:
    """Defines the input state for the storybook system."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """Messages tracking the primary execution state of the system."""
    
    project_id: str = ""
    """Unique identifier for the novel project."""
    
    phase: str = "initialization"
    """Current phase of the novel writing process."""
    
    task: str = ""
    """Current task being processed."""


@dataclass
class NovelSystemState(InputState):
    """Represents the complete state of the storybook system."""

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one before the graph raises an error."""
    
    project: ProjectInfo = field(default_factory=ProjectInfo)
    """Information about the current novel project."""
    
    current_input: Dict[str, Any] = field(default_factory=dict)
    """Current input being processed by an agent."""
    
    current_agent: str = ""
    """The current agent processing the input."""
    
    agent_outputs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    """Outputs from various agents, organized by agent name."""
    
    phase_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    """Historical record of activities in each phase."""
EOF

cat > storybook/configuration.py << 'EOF'
"""Define the configurable parameters for the storybook system."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Dict, Optional, Any

from langchain_core.runnables import RunnableConfig, ensure_config


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
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agents' interactions."
        },
    )

    mongodb_connection_string: str = field(
        default="mongodb://localhost:27017",
        metadata={
            "description": "Connection string for MongoDB checkpointing."
        },
    )
    
    mongodb_database_name: str = field(
        default="storybook_system",
        metadata={
            "description": "MongoDB database name for checkpointing."
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
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
EOF

cat > storybook/utils.py << 'EOF'
"""Utility functions for the storybook system."""

from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from datetime import datetime


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
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


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
EOF

cat > storybook/mongodb.py << 'EOF'
"""MongoDB integration for the storybook system."""

from typing import Dict, Any, Optional
from pymongo import MongoClient
from datetime import datetime


class MongoDBManager:
    """Manager for MongoDB operations."""
    
    def __init__(self, connection_string: str, database_name: str):
        """Initialize the MongoDB manager.
        
        Args:
            connection_string: MongoDB connection string.
            database_name: Name of the database to use.
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
    
    def save_project(self, project_id: str, project_data: Dict[str, Any]) -> None:
        """Save project data to MongoDB.
        
        Args:
            project_id: ID of the project.
            project_data: Project data to save.
        """
        collection = self.db.projects
        project_data["_id"] = project_id
        collection.replace_one({"_id": project_id}, project_data, upsert=True)
    
    def load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load project data from MongoDB.
        
        Args:
            project_id: ID of the project.
            
        Returns:
            Project data if found, None otherwise.
        """
        collection = self.db.projects
        return collection.find_one({"_id": project_id})
    
    def save_agent_output(self, project_id: str, agent_name: str, output: Dict[str, Any]) -> None:
        """Save agent output to MongoDB.
        
        Args:
            project_id: ID of the project.
            agent_name: Name of the agent.
            output: Agent output data.
        """
        collection = self.db.agent_outputs
        output["project_id"] = project_id
        output["agent_name"] = agent_name
        output["timestamp"] = output.get("timestamp", datetime.now().isoformat())
        collection.insert_one(output)
EOF

cat > storybook/agents.py << 'EOF'
"""Define the agents for the storybook system."""

from typing import Dict, Any, List, Callable, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from datetime import datetime

from storybook.configuration import Configuration
from storybook.state import NovelSystemState
from storybook.utils import load_chat_model


class AgentFactory:
    """Factory for creating specialized novel writing agents."""
    
    def __init__(self, config: Configuration):
        """Initialize the agent factory.
        
        Args:
            config: System configuration.
        """
        self.config = config
        self.base_model = load_chat_model(config.model)
        self.agent_roles = config.agent_roles
    
    def create_agent(self, agent_name: str, project_id: str) -> Callable:
        """Create an agent function for the specified role.
        
        Args:
            agent_name: Name/role of the agent.
            project_id: ID of the project.
            
        Returns:
            Agent function that processes state.
        """
        if agent_name not in self.agent_roles:
            raise ValueError(f"Unknown agent role: {agent_name}")
        
        role_description = self.agent_roles[agent_name]
        
        async def agent_function(state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
            """The agent function that processes state and returns an updated state.
            
            Args:
                state: Current system state.
                config: Runtime configuration.
                
            Returns:
                Dictionary with updates to the state.
            """
            configuration = Configuration.from_runnable_config(config)
            
            # Prepare the system prompt
            system_prompt = f"{configuration.system_prompt}\n\nYou are the {agent_name}. {role_description}\n\nProject ID: {project_id}\nCurrent Phase: {state.phase}\nTask: {state.current_input.get('task', '')}"
            
            # Prepare context from project data
            project_context = f"Project title: {state.project.title}\nGenre: {', '.join(state.project.genre)}\nTarget audience: {', '.join(state.project.target_audience)}"
            
            # Get the model's response
            model = self.base_model
            response = await model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    *state.messages,
                    AIMessage(content=f"Project context:\n{project_context}\n\nPlease address the current task: {state.current_input.get('task', '')}")
                ],
                config
            )
            
            # Update the state
            return {
                "messages": [response],
                "current_agent": agent_name,
                "agent_outputs": {
                    **state.agent_outputs,
                    agent_name: state.agent_outputs.get(agent_name, []) + [{
                        "timestamp": datetime.now().isoformat(),
                        "task": state.current_input.get("task", ""),
                        "response": response.content
                    }]
                }
            }
        
        return agent_function
EOF

cat > storybook/graph.py << 'EOF'
"""Define the workflow graphs for the storybook system."""

from typing import Dict, Any, List, Literal, cast
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.mongo import MongoDBCheckpointHandler

from storybook.configuration import Configuration
from storybook.state import NovelSystemState, InputState
from storybook.agents import AgentFactory
from storybook.utils import check_quality_gate


def create_phase_graph(phase: str, project_id: str, config: Configuration) -> StateGraph:
    """Create a workflow graph for a specific phase.
    
    Args:
        phase: The phase name.
        project_id: ID of the project.
        config: System configuration.
        
    Returns:
        A StateGraph for the specified phase.
    """
    # Create agent factory
    agent_factory = AgentFactory(config)
    
    # Define phase-specific agents
    agents = {
        "initialization": [
            "executive_director",
            "human_feedback_manager",
            "quality_assessment_director",
            "project_timeline_manager",
            "market_alignment_director"
        ],
        "development": [
            "executive_director",
            "creative_director",
            "structure_architect",
            "plot_development_specialist",
            "world_building_expert",
            "character_psychology_specialist",
            "character_voice_designer",
            "character_relationship_mapper",
            "domain_knowledge_specialist",
            "cultural_authenticity_expert",
            "market_alignment_director"
        ],
        "creation": [
            "executive_director",
            "content_development_director",
            "creative_director",
            "chapter_drafters",
            "scene_construction_specialists",
            "dialogue_crafters",
            "continuity_manager",
            "voice_consistency_monitor",
            "emotional_arc_designer",
            "domain_knowledge_specialist"
        ],
        "refinement": [
            "executive_director",
            "editorial_director",
            "creative_director",
            "market_alignment_director",
            "structural_editor",
            "character_arc_evaluator",
            "thematic_coherence_analyst",
            "prose_enhancement_specialist",
            "dialogue_refinement_expert",
            "rhythm_cadence_optimizer",
            "grammar_consistency_checker",
            "fact_verification_specialist"
        ],
        "finalization": [
            "executive_director",
            "editorial_director",
            "market_alignment_director",
            "positioning_specialist",
            "title_blurb_optimizer",
            "differentiation_strategist",
            "formatting_standards_expert"
        ]
    }
    
    if phase not in agents:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Create the graph builder
    builder = StateGraph(NovelSystemState, input=InputState, config_schema=Configuration)
    
    # Create and add all agents for this phase
    for agent_name in agents[phase]:
        agent_function = agent_factory.create_agent(agent_name, project_id)
        builder.add_node(agent_name, agent_function)
    
    # Set executive_director as the entry point
    builder.add_edge("__start__", "executive_director")
    
    # Define the routing functions based on the phase
    if phase == "initialization":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in initialization phase."""
            task = state.current_input.get("task", "").lower()
            
            if "human_feedback" in task:
                return "human_feedback_manager"
            elif "quality" in task or "assessment" in task:
                return "quality_assessment_director"
            elif "timeline" in task or "schedule" in task:
                return "project_timeline_manager"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "initialization_to_development", 
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )
                
                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "executive_director"
        
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        
        # All specialized agents return to executive director
        for agent in ["human_feedback_manager", "quality_assessment_director", 
                     "project_timeline_manager", "market_alignment_director"]:
            builder.add_edge(agent, "executive_director")
    
    elif phase == "development":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in development phase."""
            task = state.current_input.get("task", "").lower()
            
            if "creative" in task or "story" in task:
                return "creative_director"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            elif "research" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "development_to_creation", 
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )
                
                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "creative_director"
        
        def route_after_creative_director(state: NovelSystemState) -> str:
            """Route after the creative director node."""
            task = state.current_input.get("task", "").lower()
            
            if "structure" in task or "plot" in task:
                return "structure_architect"
            elif "character" in task and "psychology" in task:
                return "character_psychology_specialist"
            elif "character" in task and "voice" in task:
                return "character_voice_designer"
            elif "character" in task and "relationship" in task:
                return "character_relationship_mapper"
            elif "world" in task or "setting" in task:
                return "world_building_expert"
            else:
                return "executive_director"
        
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        
        builder.add_conditional_edges(
            "creative_director",
            route_after_creative_director
        )
        
        # Connect specialized agents to their supervisors
        for agent in ["structure_architect", "plot_development_specialist", 
                     "world_building_expert", "character_psychology_specialist",
                     "character_voice_designer", "character_relationship_mapper"]:
            builder.add_edge(agent, "creative_director")
        
        for agent in ["domain_knowledge_specialist", "cultural_authenticity_expert", 
                     "market_alignment_director"]:
            builder.add_edge(agent, "executive_director")
    
    elif phase == "creation":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in creation phase."""
            task = state.current_input.get("task", "").lower()
            
            if "content" in task or "development" in task:
                return "content_development_director"
            elif "creative" in task:
                return "creative_director"
            elif "domain" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "creation_to_refinement", 
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )
                
                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "content_development_director"
        
        def route_after_content_director(state: NovelSystemState) -> str:
            """Route after the content development director node."""
            task = state.current_input.get("task", "").lower()
            
            if "chapter" in task:
                return "chapter_drafters"
            elif "scene" in task:
                return "scene_construction_specialists"
            elif "dialogue" in task:
                return "dialogue_crafters"
            elif "continuity" in task:
                return "continuity_manager"
            elif "voice" in task or "tone" in task:
                return "voice_consistency_monitor"
            elif "emotion" in task or "arc" in task:
                return "emotional_arc_designer"
            else:
                return "executive_director"
        
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        
        builder.add_conditional_edges(
            "content_development_director",
            route_after_content_director
        )
        
        # Connect specialized agents to their supervisors
        for agent in ["chapter_drafters", "scene_construction_specialists", 
                     "dialogue_crafters", "continuity_manager",
                     "voice_consistency_monitor", "emotional_arc_designer"]:
            builder.add_edge(agent, "content_development_director")
        
        builder.add_edge("content_development_director", "executive_director")
        builder.add_edge("creative_director", "executive_director")
        builder.add_edge("domain_knowledge_specialist", "executive_director")
    
    elif phase == "refinement":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in refinement phase."""
            task = state.current_input.get("task", "").lower()
            
            if "editorial" in task or "edit" in task:
                return "editorial_director"
            elif "creative" in task:
                return "creative_director"
            elif "market" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "refinement_to_finalization", 
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )
                
                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "editorial_director"
        
        def route_after_editorial_director(state: NovelSystemState) -> str:
            """Route after the editorial director node."""
            task = state.current_input.get("task", "").lower()
            
            if "structure" in task:
                return "structural_editor"
            elif "character" in task and "arc" in task:
                return "character_arc_evaluator"
            elif "theme" in task:
                return "thematic_coherence_analyst"
            elif "prose" in task:
                return "prose_enhancement_specialist"
            elif "dialogue" in task:
                return "dialogue_refinement_expert"
            elif "rhythm" in task or "cadence" in task:
                return "rhythm_cadence_optimizer"
            elif "grammar" in task:
                return "grammar_consistency_checker"
            elif "fact" in task or "verify" in task:
                return "fact_verification_specialist"
            else:
                return "executive_director"
        
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        
        builder.add_conditional_edges(
            "editorial_director",
            route_after_editorial_director
        )
        
        # Connect specialized agents to their supervisors
        for agent in ["structural_editor", "character_arc_evaluator", 
                     "thematic_coherence_analyst", "prose_enhancement_specialist",
                     "dialogue_refinement_expert", "rhythm_cadence_optimizer",
                     "grammar_consistency_checker", "fact_verification_specialist"]:
            builder.add_edge(agent, "editorial_director")
        
        builder.add_edge("editorial_director", "executive_director")
        builder.add_edge("creative_director", "executive_director")
        builder.add_edge("market_alignment_director", "executive_director")
    
    elif phase == "finalization":
        def route_after_executive_director(state: NovelSystemState) -> str:
            """Route after the executive director node in finalization phase."""
            task = state.current_input.get("task", "").lower()
            
            if "editorial" in task:
                return "editorial_director"
            elif "market" in task:
                return "market_alignment_director"
            else:
                # Check quality gate to possibly end this phase
                gate_result = check_quality_gate(
                    "finalization_to_complete", 
                    state.project.quality_assessment,
                    {"quality_gates": config.quality_gates}
                )
                
                if gate_result["passed"]:
                    return "__end__"
                else:
                    return "editorial_director"
        
        def route_after_market_director(state: NovelSystemState) -> str:
            """Route after the market alignment director node."""
            task = state.current_input.get("task", "").lower()
            
            if "position" in task:
                return "positioning_specialist"
            elif "title" in task or "blurb" in task:
                return "title_blurb_optimizer"
            elif "different" in task or "unique" in task:
                return "differentiation_strategist"
            else:
                return "executive_director"
        
        builder.add_conditional_edges(
            "executive_director",
            route_after_executive_director
        )
        
        builder.add_conditional_edges(
            "market_alignment_director",
            route_after_market_director
        )
        
        # Connect specialized agents to their supervisors
        builder.add_edge("editorial_director", "executive_director")
        
        for agent in ["positioning_specialist", "title_blurb_optimizer", 
                     "differentiation_strategist"]:
            builder.add_edge(agent, "market_alignment_director")
        
        builder.add_edge("market_alignment_director", "executive_director")
        builder.add_edge("formatting_standards_expert", "editorial_director")
    
    # Set up MongoDB checkpointing
    checkpointer = MongoDBCheckpointHandler(
        connection_string=config.mongodb_connection_string,
        database_name=config.mongodb_database_name,
        collection_name=f"checkpoint_{phase}_{project_id}"
    )
    
    # Compile the graph with checkpointing
    graph = builder.compile()
    graph.set_checkpoint(checkpointer)
    graph.name = f"storybook - {phase.capitalize()} Phase"
    
    return graph


def create_supervisor_graph(config: Configuration) -> StateGraph:
    """Create the supervisor graph that manages phase transitions.
    
    Args:
        config: System configuration.
        
    Returns:
        A supervisor StateGraph.
    """
    builder = StateGraph(NovelSystemState, input=InputState, config_schema=Configuration)
    
    # Phase transition node
    async def phase_manager(state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Manage phase transitions."""
        configuration = Configuration.from_runnable_config(config)
        current_phase = state.phase
        
        # Check if we should transition to the next phase
        next_phase_map = {
            "initialization": "development",
            "development": "creation",
            "creation": "refinement",
            "refinement": "finalization",
            "finalization": "complete"
        }
        
        if current_phase not in next_phase_map:
            return {"phase": current_phase}
        
        # Check quality gate
        gate_name = f"{current_phase}_to_{next_phase_map[current_phase]}"
        gate_result = check_quality_gate(
            gate_name,
            state.project.quality_assessment,
            {"quality_gates": configuration.quality_gates}
        )
        
        if gate_result["passed"]:
            next_phase = next_phase_map[current_phase]
            # Record phase transition in history
            phase_history = state.phase_history.copy()
            if current_phase not in phase_history:
                phase_history[current_phase] = []
            
            phase_history[current_phase].append({
                "end_time": datetime.now().isoformat(),
                "transition_to": next_phase,
                "quality_assessment": state.project.quality_assessment
            })
            
            return {
                "phase": next_phase,
                "phase_history": phase_history,
                "messages": [
                    AIMessage(content=f"The project has successfully passed the quality gate from {current_phase} to {next_phase}. Phase transition successful.")
                ]
            }
        else:
            # Stay in current phase
            return {"phase": current_phase}
    
    builder.add_node("phase_manager", phase_manager)
    builder.set_entry_point("phase_manager")
    builder.add_edge("phase_manager", "__end__")
    
    # Compile the supervisor graph
    graph = builder.compile()
    graph.name = "storybook - Phase Supervisor"
    
    return graph


# Main export - the novel writing system graph
def create_novel_writing_system(project_id: str, config: Configuration = None) -> Dict[str, StateGraph]:
    """Create the complete storybook system with all phase graphs.
    
    Args:
        project_id: ID of the project.
        config: Optional system configuration.
        
    Returns:
        Dictionary of graphs for each phase and the supervisor.
    """
    if config is None:
        config = Configuration()
    
    return {
        "supervisor": create_supervisor_graph(config),
        "initialization": create_phase_graph("initialization", project_id, config),
        "development": create_phase_graph("development", project_id, config),
        "creation": create_phase_graph("creation", project_id, config),
        "refinement": create_phase_graph("refinement", project_id, config),
        "finalization": create_phase_graph("finalization", project_id, config)
    }
EOF

cat > storybook/__init__.py << 'EOF'
"""storybook system using a multi-agent approach.

This module defines a collaborative AI system for novel creation,
with specialized agents handling different aspects of the writing process.
"""

from storybook.graph import create_novel_writing_system
from storybook.configuration import Configuration

__all__ = ["create_novel_writing_system", "Configuration"]
EOF

# Make the script executable
chmod +x "$0"

echo "All storybook system files have been created successfully!"
