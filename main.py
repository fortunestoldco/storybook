from dotenv import load_dotenv
import os
import sys
sys.path.append("../utils")
import utils

# Load environment variables from .env file or Secret Manager
_ = load_dotenv("../.env")
aws_region = os.getenv("AWS_REGION")
tavily_ai_api_key = utils.get_tavily_api("TAVILY_API_KEY", aws_region)

import warnings
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, cast
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ChatMessage,
)

import boto3
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
import sqlite3
from datetime import datetime
from uuid import uuid4
from langchain_core.runnables import RunnableConfig


# for the output parser
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import json


# Define the project types and input states
class ProjectType:
    NEW = "new"
    EXISTING = "existing"

class NewProjectInput(TypedDict):
    title: str
    synopsis: str
    manuscript: Optional[str]
    notes: Optional[Dict[str, Any]]

class ExistingProjectInput(TypedDict):
    project_id: str

class ProjectData(TypedDict):
    id: str
    title: str
    synopsis: str
    manuscript: Optional[str]
    notes: Optional[Dict[str, Any]]
    type: str
    quality_assessment: Dict[str, Any]
    created_at: str

class InputState(TypedDict):
    project_type: str
    project_data: Dict[str, Any]
    task: str


# Define the research state classes
class ResearchState(TypedDict):
    query: str
    results: List[Dict[str, Any]]
    summary: str

class DomainResearchState(ResearchState):
    domain_specific_data: Dict[str, Any]

class CulturalResearchState(ResearchState):
    cultural_context: Dict[str, Any]

class MarketResearchState(ResearchState):
    market_trends: Dict[str, Any]

class FactVerificationState(ResearchState):
    verification_status: Dict[str, bool]


# Combined state for the StoryBook application
class AgentState(TypedDict):
    # Original essay writer states
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]
    
    # StoryBook states
    project: Optional[ProjectData]
    phase: Optional[str]
    phase_history: Optional[Dict[str, List[Dict[str, Any]]]]
    current_input: Optional[Dict[str, Any]]
    messages: Optional[List[Dict[str, Any]]]


class Queries(BaseModel):
    queries: List[str] = Field(description="List of research queries")


class Configuration(BaseModel):
    mongodb_connection_string: Optional[str] = None
    mongodb_database_name: Optional[str] = None
    quality_gates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """Extract configuration from a runnable config."""
        configurable = config.get("configurable", {})
        return cls(
            mongodb_connection_string=configurable.get("mongodb_connection_string"),
            mongodb_database_name=configurable.get("mongodb_database_name"),
            quality_gates=configurable.get("quality_gates", {})
        )


class StoryBookConfig(BaseModel):
    model_name: str
    temperature: float
    max_tokens: Optional[int] = None


def check_quality_gate(gate_name: str, quality_assessment: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a quality gate is passed."""
    gates = config.get("quality_gates", {})
    gate_config = gates.get(gate_name, {})
    
    if not gate_config:
        # If no gate is defined, default to passing
        return {"passed": True, "message": f"No quality gate defined for {gate_name}"}
    
    # Check each criterion in the gate
    passed = True
    reasons = []
    
    for criterion, threshold in gate_config.items():
        if criterion in quality_assessment:
            value = quality_assessment[criterion]
            if value < threshold:
                passed = False
                reasons.append(f"{criterion}: {value} (below threshold {threshold})")
        else:
            # If the criterion is not in the assessment, consider it failed
            passed = False
            reasons.append(f"{criterion}: not assessed (required)")
    
    return {
        "passed": passed,
        "message": "Quality gate passed" if passed else "Quality gate failed",
        "reasons": reasons
    }


class AgentFactory:
    def __init__(self, config, bedrock_client=None, tavily_client=None):
        self.config = config
        self.bedrock_rt = bedrock_client or boto3.client("bedrock-runtime", region_name=aws_region)
        self.tavily = tavily_client or TavilyClient(api_key=tavily_ai_api_key)
        self.model = ChatBedrockConverse(
            client=self.bedrock_rt,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0,
            max_tokens=None,
        )
    
    def create_agent(self, agent_name: str, project_id: str):
        """Create a function for a specific agent."""
        agent_prompts = {
            "executive_director": """You are the Executive Director responsible for overseeing the entire novel creation process. 
            Your role is to coordinate all aspects of the project, maintain the vision, and ensure quality standards are met.""",
            
            "creative_director": """You are the Creative Director responsible for the artistic vision of the novel.
            Your role is to ensure creative consistency, develop the narrative arc, and maintain the novel's artistic integrity.""",
            
            "structure_architect": """You are the Structure Architect responsible for designing the novel's structure.
            Your role is to create a compelling narrative framework that supports the story.""",
            
            "character_psychology_specialist": """You are the Character Psychology Specialist who ensures characters have 
            depth, realistic motivations, and consistent psychological profiles.""",
            
            "world_building_expert": """You are the World Building Expert who creates immersive, coherent and believable
            settings for the story to take place in.""",
            
            "editorial_director": """You are the Editorial Director who oversees revisions, edits and quality control
            processes to ensure the novel meets professional standards.""",
            
            "market_alignment_director": """You are the Market Alignment Director who ensures the novel
            appeals to its target audience and meets market expectations.""",
        }
        
        prompt = agent_prompts.get(agent_name, f"You are the {agent_name} responsible for your specialized role in novel creation.")
        
        def agent_function(state: AgentState):
            current_task = state.get("current_input", {}).get("task", state.get("task", ""))
            current_phase = state.get("phase", "initialization")
            project = state.get("project", {})
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"""
                Project: {project.get('title', 'Untitled')}
                Synopsis: {project.get('synopsis', 'No synopsis provided')}
                Current Phase: {current_phase}
                Task: {current_task}
                """)
            ]
            
            # Add relevant context from the project
            if project and project.get("notes"):
                context = "\n\nAdditional context:\n"
                for key, value in project["notes"].items():
                    context += f"{key}: {value}\n"
                messages[1].content += context
            
            response = self.model.invoke(messages)
            
            # Update the quality assessment based on the agent's contribution
            quality_assessment = {}
            if project:
                quality_assessment = project.get("quality_assessment", {}).copy()
            
            # Different agents contribute to different quality metrics
            if agent_name == "structure_architect":
                quality_assessment["structure_quality"] = 0.8  # Example value
            elif agent_name == "character_psychology_specialist":
                quality_assessment["character_depth"] = 0.9  # Example value
            
            # Update both essay writing and storybook states
            result = {
                "lnode": agent_name,
                "count": 1
            }
            
            if project:
                result["project"] = {
                    **project,
                    "quality_assessment": quality_assessment
                }
                result["messages"] = state.get("messages", []) + [{"role": "assistant", "content": response.content}]
            
            # Add agent's response to the appropriate field based on its function
            if agent_name == "structure_architect":
                result["plan"] = response.content
            elif "draft" in agent_name.lower() or "content" in agent_name.lower():
                result["draft"] = response.content
            elif "edit" in agent_name.lower() or "critique" in agent_name.lower():
                result["critique"] = response.content
            
            return result
        
        return agent_function


def create_research_subgraph(research_type: str, state_class: type, config: Dict[str, Any]) -> StateGraph:
    """Create a research subgraph for a specific research type."""
    builder = StateGraph(state_class)
    
    def generate_queries(state: Dict[str, Any]):
        """Generate research queries based on the current state."""
        query = state.get("query", "")
        return {"query": query}
    
    def conduct_research(state: Dict[str, Any]):
        """Conduct research using Tavily."""
        query = state.get("query", "")
        tavily = TavilyClient(api_key=tavily_ai_api_key)
        results = tavily.search(query=query, max_results=3)
        return {"results": results["results"]}
    
    def summarize_results(state: Dict[str, Any]):
        """Summarize research results."""
        results = state.get("results", [])
        content = "\n\n".join([r.get("content", "") for r in results])
        
        bedrock_rt = boto3.client("bedrock-runtime", region_name=aws_region)
        model = ChatBedrockConverse(
            client=bedrock_rt,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0,
            max_tokens=None,
        )
        
        messages = [
            SystemMessage(content=f"Summarize the following research information for {research_type} research:"),
            HumanMessage(content=content)
        ]
        
        response = model.invoke(messages)
        
        # Specific processing based on research type
        if research_type == "domain":
            return {
                "summary": response.content,
                "domain_specific_data": {"source": "research", "content": response.content}
            }
        elif research_type == "cultural":
            return {
                "summary": response.content,
                "cultural_context": {"source": "research", "content": response.content}
            }
        elif research_type == "market":
            return {
                "summary": response.content,
                "market_trends": {"source": "research", "content": response.content}
            }
        elif research_type == "fact":
            return {
                "summary": response.content,
                "verification_status": {"accurate": True}  # Simplified
            }
        else:
            return {"summary": response.content}
    
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("summarize_results", summarize_results)
    
    builder.set_entry_point("generate_queries")
    builder.add_edge("generate_queries", "conduct_research")
    builder.add_edge("conduct_research", "summarize_results")
    builder.add_edge("summarize_results", END)
    
    return builder.compile()


def create_phase_graph(phase: str, project_id: str, config: Dict[str, Any]) -> StateGraph:
    """Create a workflow graph for a specific phase."""
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
    builder = StateGraph(AgentState)
    
    # Create research agents dictionary
    research_agents = {
        "domain_knowledge_specialist": {
            "research_type": "domain",
            "state_class": DomainResearchState
        },
        "cultural_authenticity_expert": {
            "research_type": "cultural", 
            "state_class": CulturalResearchState
        },
        "market_alignment_director": {
            "research_type": "market",
            "state_class": MarketResearchState
        },
        "fact_verification_specialist": {
            "research_type": "fact",
            "state_class": FactVerificationState
        }
    }
    
    # Create and add all agents for this phase
    for agent_name in agents[phase]:
        if agent_name in research_agents:
            # For simplicity in the AWS sample, we'll create regular agents instead of subgraphs
            agent_function = agent_factory.create_agent(agent_name, project_id)
            builder.add_node(agent_name, agent_function)
        else:
            # Create regular agent node
            agent_function = agent_factory.create_agent(agent_name, project_id)
            builder.add_node(agent_name, agent_function)
    
    # Set executive_director as the entry point
    builder.add_edge("__start__", "executive_director")
    
    # Define the routing functions based on the phase
    if phase == "initialization":
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in initialization phase."""
            task = state.get("current_input", {}).get("task", state.get("task", "")).lower()

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
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "initialization_to_development",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )

                if gate_result["passed"]:
                    return END
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
        def route_after_executive_director(state: AgentState) -> str:
            """Route after the executive director node in development phase."""
            task = state.get("current_input", {}).get("task", state.get("task", "")).lower()

            if "creative" in task or "story" in task:
                return "creative_director"
            elif "market" in task or "trend" in task:
                return "market_alignment_director"
            elif "research" in task or "knowledge" in task:
                return "domain_knowledge_specialist"
            else:
                # Check quality gate to possibly end this phase
                project = state.get("project", {})
                quality_assessment = project.get("quality_assessment", {})
                gate_result = check_quality_gate(
                    "development_to_creation",
                    quality_assessment,
                    {"quality_gates": config.get("quality_gates", {})}
                )

                if gate_result["passed"]:
                    return END
                else:
                    return "creative_director"

        def route_after_creative_director(state: AgentState) -> str:
            """Route after the creative director node."""
            task = state.get("current_input", {}).get("task", state.get("task", "")).lower()

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
            if agent in agents[phase]:  # Only add edges for agents that exist in this phase
                builder.add_edge(agent, "creative_director")

        for agent in ["domain_knowledge_specialist", "cultural_authenticity_expert",
                     "market_alignment_director"]:
            if agent in agents[phase]:  # Only add edges for agents that exist in this phase
                builder.add_edge(agent, "executive_director")
    
    # Similar implementation for other phases would go here
    
    return builder.compile()


class ewriter:
    def __init__(self):

        self.bedrock_rt = boto3.client("bedrock-runtime", region_name=aws_region)
        self.tavily = TavilyClient(api_key=tavily_ai_api_key)
        self.model = ChatBedrockConverse(
            client=self.bedrock_rt,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0,
            max_tokens=None,
        )

        # self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.PLAN_PROMPT = (
            "You are an expert writer tasked with writing a high level outline of a short 3 paragraph essay. "
            "Write such an outline for the user provided topic. Give the three main headers of an outline of "
            "the essay along with any relevant notes or instructions for the sections. "
        )
        self.WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
            Generate the best essay possible for the user's request and the initial outline. \
            If the user provides critique, respond with a revised version of your previous attempts. \
            Utilize all the information below as needed: 

            ------
            <content>
            {content}
            </content>"""
        self.RESEARCH_PLAN_PROMPT = (
            "You are a researcher charged with providing information that can "
            "be used when writing the following essay. Generate a list of search "
            "queries that will gather "
            "any relevant information. Only generate 3 queries max."
        )
        self.REFLECTION_PROMPT = (
            "You are a teacher grading an 3 paragraph essay submission. "
            "Generate critique and recommendations for the user's submission. "
            "Provide detailed recommendations, including requests for length, depth, style, etc."
        )
        self.RESEARCH_CRITIQUE_PROMPT = (
            "You are a researcher charged with providing information that can "
            "be used when making any requested revisions (as outlined below). "
            "Generate a list of search queries that will gather any relevant information. "
            "Only generate 2 queries max."
        )
        
        # Setup StoryBook configuration
        self.storybook_config = {
            "mongodb_connection_string": None,  # No MongoDB for AWS sample
            "mongodb_database_name": None,
            "quality_gates": {
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
        }
        
        # Build the original essay writing graph
        self.build_essay_graph()
        
        # Build the StoryBook graph for the initialization phase
        self.phase_graphs = {}
        for phase in ["initialization", "development", "creation", "refinement", "finalization"]:
            self.phase_graphs[phase] = create_phase_graph(phase, "default_project", self.storybook_config)

    def build_essay_graph(self):
        """Build the original essay writing graph"""
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", self.should_continue, {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=[
                "planner",
                "generate",
                "reflect",
                "research_plan",
                "research_critique",
            ],
        )

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ]
        response = self.model.invoke(messages)
        return {
            "plan": response.content,
            "lnode": "planner",
            "count": 1,
        }

    def research_plan_node(self, state: AgentState):
        # Set up the Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=Queries)

        # Create a prompt template with format instructions
        prompt = PromptTemplate(
            template="Generate research queries based on the given task.\n{format_instructions}\nTask: {task}\n",
            input_variables=["task"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Use the model with the new prompt and parser
        queries_output = self.model.invoke(prompt.format_prompt(task=state["task"]))

        # Extract the content from the AIMessage
        queries_text = queries_output.content

        # Extract the JSON string from the content
        json_start = queries_text.find("{")
        json_end = queries_text.rfind("}") + 1
        json_str = queries_text[json_start:json_end]

        # Parse the JSON string
        queries_dict = json.loads(json_str)

        # Create a Queries object from the parsed JSON
        parsed_queries = Queries(**queries_dict)

        content = state["content"] or []
        for q in parsed_queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {
            "content": content,
            "queries": parsed_queries.queries,
            "lnode": "research_plan",
            "count": 1,
        }

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state["content"] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        )
        messages = [
            SystemMessage(content=self.WRITER_PROMPT.format(content=content)),
            user_message,
        ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
            "lnode": "generate",
            "count": 1,
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state["draft"]),
        ]
        response = self.model.invoke(messages)
        return {
            "critique": response.content,
            "lnode": "reflect",
            "count": 1,
        }

    def research_critique_node(self, state: AgentState):
        # Set up the Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=Queries)

        # Create a prompt template with format instructions
        prompt = PromptTemplate(
            template="Generate research queries based on the given critique.\n{format_instructions}\nCritique: {critique}\n",
            input_variables=["critique"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Use the model with the new prompt and parser
        queries_output = self.model.invoke(
            prompt.format_prompt(critique=state["critique"])
        )

        # Extract the content from the AIMessage
        queries_text = queries_output.content

        # Extract the JSON string from the content
        json_start = queries_text.find("{")
        json_end = queries_text.rfind("}") + 1
        json_str = queries_text[json_start:json_end]

        # Parse the JSON string
        queries_dict = json.loads(json_str)

        # Create a Queries object from the parsed JSON
        parsed_queries = Queries(**queries_dict)

        content = state["content"] or []
        for q in parsed_queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {
            "content": content,
            "lnode": "research_critique",
            "count": 1,
        }

    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"
    
    def initialize_storybook_project(self, title, synopsis, task=None):
        """Initialize a new StoryBook project"""
        project_id = str(uuid4())
        
        initial_state = {
            "project": {
                "id": project_id,
                "title": title,
                "synopsis": synopsis,
                "manuscript": "",
                "notes": {},
                "type": "creation",
                "quality_assessment": {
                    "planning_quality": 0.6,  # Initial values
                    "market_alignment": 0.5,
                },
                "created_at": datetime.now().isoformat()
            },
            "phase": "initialization",
            "current_input": {
                "task": task or "Initial project assessment and planning",
                "phase": "initialization"
            },
            "phase_history": {},
            "messages": [],
            "count": 0,
            "revision_number": 0,
            "max_revisions": 3,
            "lnode": "",
            "task": task or "Initial project assessment and planning"
        }
        
        return initial_state
    
    def run_storybook_phase(self, state, phase="initialization"):
        """Run a specific phase of the StoryBook workflow"""
        if phase not in self.phase_graphs:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Make sure the state includes the phase
        if "phase" not in state:
            state["phase"] = phase
            
        # Run the phase graph
        result = self.phase_graphs[phase].invoke(state)
        return result


import gradio as gr
import time


class writer_gui:
    def __init__(self, graph):
        self.graph = graph
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        # self.sdisps = {} #global
        self.demo = self.create_interface()

    def run_agent(self, start, topic, stop_after):
        # global partial_message, thread_id,thread
        # global response, max_iterations, iterations, threads
        if start:
            self.iterations.append(0)
            config = {
                "task": topic,
                "max_revisions": 2,
                "revision_number": 0,
                "lnode": "",
                "planner": "no plan",
                "draft": "no draft",
                "critique": "no critique",
                "content": [
                    "no content",
                ],
                "queries": "no queries",
                "count": 0,
            }
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            ## fix
            lnode, nnode, _, rev, acount = self.get_disp_state()
            yield self.partial_message, lnode, nnode, self.thread_id, rev, acount
            config = None  # need
            # print(f"run_agent:{lnode}")
            if not nnode:
                # print("Hit the end")
                return
            if lnode in stop_after:
                # print(f"stopping due to stop_after {lnode}")
                return
            else:
                # print(f"Not stopping on lnode {lnode}")
                pass
        return
    
    def run_storybook(self, title, synopsis, task, phase, max_iterations=5):
        """Run the StoryBook workflow"""
        # Initialize a new project
        initial_state = self.graph.initialize_storybook_project(title, synopsis, task)
        
        # Create a new thread
        self.iterations.append(0)
        self.thread_id += 1
        self.threads.append(self.thread_id)
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        
        # Clear previous output
        self.partial_message = ""
        
        # Run the selected phase (simplified for demo)
        iterations = 0
        state = initial_state
        
        while iterations < max_iterations:
            try:
                result = self.graph.run_storybook_phase(state, phase)
                iterations += 1
                
                # Update the state for the next iteration
                state = result
                
                # Update the output
                self.partial_message += f"Iteration {iterations}:\n"
                self.partial_message += f"Phase: {state.get('phase')}\n"
                self.partial_message += f"Last Agent: {state.get('lnode')}\n"
                
                # Add agent messages if available
                if state.get('messages'):
                    last_message = state['messages'][-1]
                    if isinstance(last_message, dict) and 'content' in last_message:
                        self.partial_message += f"Message: {last_message['content'][:200]}...\n"
                
                self.partial_message += f"\n------------------\n\n"
                
                # Extract state information for the UI
                lnode = state.get('lnode', '')
                project = state.get('project', {})
                quality = project.get('quality_assessment', {})
                quality_str = ", ".join([f"{k}: {v}" for k, v in quality.items()])
                
                yield self.partial_message, lnode, phase, self.thread_id, quality_str
                
                # Check if we should end this phase
                if state.get('phase') != phase:
                    self.partial_message += f"Phase transition detected to {state.get('phase')}\n"
                    yield self.partial_message, lnode, state.get('phase'), self.thread_id, quality_str
                    break
                    
            except Exception as e:
                self.partial_message += f"Error: {str(e)}\n"
                yield self.partial_message, "error", phase, self.thread_id, ""
                break
        
        return

    def get_disp_state(
        self,
    ):
        current_state = self.graph.get_state(self.thread)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        # print  (lnode,nnode,self.thread_id,rev,acount)
        return lnode, nnode, self.thread_id, rev, acount

    def get_state(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            lnode, nnode, self.thread_id, rev, astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""

    def get_content(
        self,
    ):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            lnode, nnode, thread_id, rev, astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {thread_id}, rev: {rev}, step: {astep}"
            return gr.update(
                label=new_label, value="\n\n".join(item for item in content) + "\n\n"
            )
        else:
            return ""

    def update_hist_pd(
        self,
    ):
        # print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata["step"] < 1:
                continue
            thread_ts = state.config["configurable"]["thread_ts"]
            tid = state.config["configurable"]["thread_id"]
            count = state.values["count"]
            lnode = state.values["lnode"]
            rev = state.values["revision_number"]
            nnode = state.next
            st = f"{tid}:{count}:{lnode}:{nnode}:{rev}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(
            label="update_state from: thread:count:last_node:next_node:rev:thread_ts",
            choices=hist,
            value=hist[0],
            interactive=True,
        )

    def find_config(self, thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config["configurable"]["thread_ts"] == thread_ts:
                return config
        return None

    def copy_state(self, hist_str):
        """result of selecting an old state from the step pulldown. Note does not change thread.
        This copies an old state to a new current state.
        """
        thread_ts = hist_str.split(":")[-1]
        # print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        # print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(
            self.thread, state.values, as_node=state.values["lnode"]
        )
        new_state = self.graph.get_state(self.thread)  # should now match
        new_thread_ts = new_state.config["configurable"]["thread_ts"]
        tid = new_state.config["configurable"]["thread_id"]
        count = new_state.values["count"]
        lnode = new_state.values["lnode"]
        rev = new_state.values["revision_number"]
        nnode = new_state.next
        return lnode, nnode, new_thread_ts, rev, count

    def update_thread_pd(
        self,
    ):
        # print("update_thread_pd")
        return gr.Dropdown(
            label="choose thread",
            choices=threads,
            value=self.thread_id,
            interactive=True,
        )

    def switch_thread(self, new_thread_id):
        # print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return

    def modify_state(self, key, asnode, new_state):
        """gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        """
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        return

    def create_interface(self):
        with gr.Blocks(
            theme=gr.themes.Default(spacing_size="sm", text_size="sm"),
            analytics_enabled=False 
        ) as demo:

            def updt_disp():
                """general update display on state change"""
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata["step"] < 1:  # ignore early states
                        continue
                    s_thread_ts = state.config["configurable"]["thread_ts"]
                    s_tid = state.config["configurable"]["thread_id"]
                    s_count = state.values["count"]
                    s_lnode = state.values["lnode"]
                    s_rev = state.values["revision_number"]
                    s_nnode = state.next
                    st = f"{s_tid}:{s_count}:{s_lnode}:{s_nnode}:{s_rev}:{s_thread_ts}"
                    hist.append(st)
                if not current_state.metadata:  # handle init call
                    return {}
                else:
                    return {
                        topic_bx: current_state.values["task"],
                        lnode_bx: current_state.values["lnode"],
                        count_bx: current_state.values["count"],
                        revision_bx: current_state.values["revision_number"],
                        nnode_bx: current_state.next,
                        threadid_bx: self.thread_id,
                        thread_pd: gr.Dropdown(
                            label="choose thread",
                            choices=self.threads,
                            value=self.thread_id,
                            interactive=True,
                        ),
                        step_pd: gr.Dropdown(
                            label="update_state from: thread:count:last_node:next_node:rev:thread_ts",
                            choices=hist,
                            value=hist[0],
                            interactive=True,
                        ),
                    }

            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ["plan", "draft", "critique"]:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if "content" in state.values:
                        for i in range(len(state.values["content"])):
                            state.values["content"][i] = (
                                state.values["content"][i][:20] + "..."
                            )
                    if "writes" in state.metadata:
                        state.metadata["writes"] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                # print(f"vary_btn{stat}")
                return gr.update(variant=stat)

            with gr.Tab("Essay Agent"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="Essay Topic", value="Pizza Shop")
                    gen_btn = gr.Button(
                        "Generate Essay", scale=0, min_width=80, variant="primary"
                    )
                    cont_btn = gr.Button("Continue Essay", scale=0, min_width=80)
                with gr.Row():
                    lnode_bx = gr.Textbox(label="last node", min_width=100)
                    nnode_bx = gr.Textbox(label="next node", min_width=100)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80)
                    revision_bx = gr.Textbox(label="Draft Rev", scale=0, min_width=80)
                    count_bx = gr.Textbox(label="count", scale=0, min_width=80)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove("__start__")
                    stop_after = gr.CheckboxGroup(
                        checks,
                        label="Interrupt After State",
                        value=checks,
                        scale=0,
                        min_width=400,
                    )
                    with gr.Row():
                        thread_pd = gr.Dropdown(
                            choices=self.threads,
                            interactive=True,
                            label="select thread",
                            min_width=120,
                            scale=0,
                        )
                        step_pd = gr.Dropdown(
                            choices=["N/A"],
                            interactive=True,
                            label="select step",
                            min_width=160,
                            scale=1,
                        )
                live = gr.Textbox(label="Live Agent Output", lines=5, max_lines=5)

                # actions
                sdisps = [
                    topic_bx,
                    lnode_bx,
                    nnode_bx,
                    threadid_bx,
                    revision_bx,
                    count_bx,
                    step_pd,
                    thread_pd,
                ]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps
                )
                step_pd.input(self.copy_state, [step_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps
                )
                gen_btn.click(
                    vary_btn, gr.Number("secondary", visible=False), gen_btn
                ).then(
                    fn=self.run_agent,
                    inputs=[gr.Number(True, visible=False), topic_bx, stop_after],
                    outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx],
                    show_progress=True,
                ).then(
                    fn=updt_disp, inputs=None, outputs=sdisps
                ).then(
                    vary_btn, gr.Number("primary", visible=False), gen_btn
                ).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn
                )
                cont_btn.click(
                    vary_btn, gr.Number("secondary", visible=False), cont_btn
                ).then(
                    fn=self.run_agent,
                    inputs=[gr.Number(False, visible=False), topic_bx, stop_after],
                    outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx],
                ).then(
                    fn=updt_disp, inputs=None, outputs=sdisps
                ).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn
                )

            with gr.Tab("StoryBook Agent"):
                with gr.Row():
                    title_bx = gr.Textbox(label="Story Title", value="The Hidden Quest")
                    synopsis_bx = gr.Textbox(label="Synopsis", value="A young adventurer discovers a mysterious artifact that leads to an unexpected journey.", lines=3)
                with gr.Row():
                    task_bx = gr.Textbox(label="Task", value="Initial project assessment and planning")
                    phase_select = gr.Dropdown(
                        ["initialization", "development", "creation", "refinement", "finalization"],
                        label="Phase",
                        value="initialization"
                    )
                    storybook_btn = gr.Button("Run StoryBook", variant="primary")
                with gr.Row():
                    sb_lnode_bx = gr.Textbox(label="Last Agent", min_width=100)
                    sb_phase_bx = gr.Textbox(label="Current Phase", min_width=100)
                    sb_thread_bx = gr.Textbox(label="Thread", scale=0, min_width=80)
                    sb_quality_bx = gr.Textbox(label="Quality Assessment", min_width=200)
                sb_live = gr.Textbox(label="StoryBook Output", lines=10)
                
                # StoryBook actions
                storybook_btn.click(
                    vary_btn, gr.Number("secondary", visible=False), storybook_btn
                ).then(
                    fn=self.run_storybook,
                    inputs=[title_bx, synopsis_bx, task_bx, phase_select],
                    outputs=[sb_live, sb_lnode_bx, sb_phase_bx, sb_thread_bx, sb_quality_bx],
                    show_progress=True
                ).then(
                    vary_btn, gr.Number("primary", visible=False), storybook_btn
                )

            with gr.Tab("Plan"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                plan = gr.Textbox(label="Plan", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("plan", visible=False),
                    outputs=plan,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.Number("plan", visible=False),
                        gr.Number("planner", visible=False),
                        plan,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("Research Content"):
                refresh_btn = gr.Button("Refresh")
                content_bx = gr.Textbox(label="content", lines=10)
                refresh_btn.click(fn=self.get_content, inputs=None, outputs=content_bx)
            with gr.Tab("Draft"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                draft_bx = gr.Textbox(label="draft", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("draft", visible=False),
                    outputs=draft_bx,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.Number("draft", visible=False),
                        gr.Number("generate", visible=False),
                        draft_bx,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("Critique"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                critique_bx = gr.Textbox(label="Critique", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("critique", visible=False),
                    outputs=critique_bx,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.Number("critique", visible=False),
                        gr.Number("reflect", visible=False),
                        critique_bx,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("StateSnapShots"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                snapshots = gr.Textbox(label="State Snapshots Summaries")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
        return demo

    def launch(self):
            self.demo.launch(share=True)


if __name__ == "__main__":
    MultiAgent = ewriter()
    app = writer_gui(MultiAgent.graph)
    app.launch()
