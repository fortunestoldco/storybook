from typing import Dict, Any, Type, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph_sdk.clients import SyncAssistantsClient
from langchain_core.agents import AgentExecutor
from langchain_core.tools import BaseTool

from mongodb import MongoDBManager
from config import MODEL_CONFIGS
from state import NovelSystemState
from tools.development import get_tools_for_agent

class AgentFactory:
    """Factory for creating and managing agents in the LangGraph workflow."""
    
    def __init__(self, mongodb_manager: MongoDBManager):
        self.mongodb_manager = mongodb_manager
        self._agent_cache: Dict[str, Any] = {}
        self.assistants_client = SyncAssistantsClient()
        self.phase_agents = {
            "initialization": ["quality_assessment_director", "plot_development", "character_development"],
            "development": ["creative_director", "structure_architect", "world_building_expert"],
            "creation": ["content_creator", "draft_reviewer", "continuity_manager"],
            "refinement": ["editorial_director", "prose_enhancement_specialist", "grammar_consistency_checker"],
            "finalization": ["market_alignment_director", "finalizer", "quality_checker"]
        }
        
    def get_phase_agents(self, phase: str) -> list[Callable]:
        """Get all agent functions for a specific phase."""
        agent_types = self.phase_agents.get(phase, [])
        return [self.create_workflow_agent(agent_type) for agent_type in agent_types]
    
    def create_phase_subgraph(self, phase: str, state_type: Type[NovelSystemState]) -> StateGraph:
        """Create a subgraph for a specific workflow phase."""
        graph = StateGraph(state_type)
        
        # Add all agents for this phase
        for agent_type in self.phase_agents.get(phase, []):
            agent_function = self.create_workflow_agent(agent_type)
            graph.add_node(
                agent_type,
                agent_function,
                metadata={"phase": phase, "agent_type": agent_type}
            )
        
        # Connect agents in sequence
        agents = self.phase_agents.get(phase, [])
        for i in range(len(agents) - 1):
            graph.add_edge(agents[i], agents[i + 1])
            
        if agents:
            graph.add_edge(agents[-1], END)
            
        return graph.compile()
        
    def create_workflow_agent(
        self,
        agent_type: str,
        state_type: Type[NovelSystemState],
        tools: Optional[list[BaseTool]] = None
    ) -> Callable:
        """Create a workflow-compatible agent function.
        
        Args:
            agent_type: Type of agent to create
            state_type: Type of state the agent handles
            tools: Optional list of tools for the agent
            
        Returns:
            A callable agent function for use in the workflow
        """
        def agent_function(state: Dict) -> Dict:
                }
                
            except Exception as e:
                return {
                    **state,
                    "errors": state.get("errors", []) + [{
                        "agent": agent_type,
                        "error": str(e)
                    }]
                }
                
        return agent_function
    
    def create_graph_node(
        self,
        agent_type: str,
        state_type: Type[NovelSystemState],
        next_nodes: Optional[list[str]] = None
    ) -> StateGraph:
        """Create a workflow graph node for an agent.
        
        Args:
            agent_type: Type of agent for this node
            state_type: Type of state the node handles
            next_nodes: List of nodes that can follow this one
            
        Returns:
            A StateGraph node
        """
        # Get tools for this agent type
        tools = get_tools_for_agent(agent_type)
        tool_executor = ToolExecutor(tools=tools)
        
        # Create the agent function
        agent_function = self.create_workflow_agent(
            agent_type=agent_type,
            state_type=state_type,
            tools=tools
        )
        
        # Create graph node
        graph = StateGraph(state_type)
        
        # Add the agent node
        graph.add_node(
            agent_type,
            agent_function,
            metadata={
                "description": MODEL_CONFIGS[agent_type].get("description", ""),
                "agent_type": agent_type
            }
        )
        
        # Add edges
        if next_nodes:
            for next_node in next_nodes:
                graph.add_edge(agent_type, next_node)
        else:
            graph.add_edge(agent_type, END)
            
        return graph.compile()
    
    def _get_or_create_assistant(self, agent_type: str) -> str:
        """Get or create a LangGraph assistant for an agent type."""
        cache_key = f"assistant_{agent_type}"
        
        if cache_key not in self._agent_cache:
            config = MODEL_CONFIGS.get(agent_type, {})
            
            # Create assistant
            assistant = self.assistants_client.create(
                name=f"{agent_type}_assistant",
                model=config.get("model", "anthropic/claude-3-opus-20240229"),
                instructions=config.get("prompt_template", ""),
                tools=get_tools_for_agent(agent_type)
            )
            
            self._agent_cache[cache_key] = assistant.id
            
        return self._agent_cache[cache_key]
