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
            try:
                # For cloud deployment, use Assistants API
                if os.getenv("LANGGRAPH_CLOUD") >= "true":
                    assistant_id = self._get_or_create_assistant(agent_type)
                    
                    def cloud_agent_function(state: NovelSystemState) -> Dict:
                        """The agent function to be used in the graph with LangGraph Cloud.

                        Args:
                            state: The current state.

                        Returns:
                            The updated state.
                        """
                        try:
                            # Prepare the context
                            context = {
                                "project_state": json.dumps(state["project"], indent=2),
                                "current_phase": state["project"].current_phase,
                                "task": state["current_input"].get("task", ""),
                                "input": state["current_input"].get("content", "")
                            }
                            
                            # Create a thread and run the assistant
                            thread = self.assistants_client.create_thread()
                            self.assistants_client.add_message(
                                thread_id=thread.id,
                                role="user",
                                content=json.dumps(context)
                            )
                            run = self.assistants_client.run_thread(
                                thread_id=thread.id,
                                assistant_id=assistant_id
                            )
                            
                            # Get the response
                            messages = self.assistants_client.get_messages(thread.id)
                            response = messages[0].content[0].text.value
                            
                            # Update the state
                            state["current_output"] = {
                                "agent": agent_type,
                                "content": response,
                                "timestamp": current_timestamp()
                            }

                            # Add to message history
                            state["messages"].append({
                                "role": agent_type,
                                "content": response,
                                "timestamp": current_timestamp()
                            })

                            return state
                        except Exception as e:
                            logger.error(f"Error in cloud agent function for agent {agent_type}: {e}")
                            state["errors"].append({
                                "agent": agent_type,
                                "error": str(e),
                                "timestamp": current_timestamp()
                            })
                            return state
                    
                    return cloud_agent_function
                
                # For local deployment, use LangChain
                else:
                    llm = self._get_llm(agent_type)
                    memory = self._get_memory(agent_type, project_id)
                    prompt = self._get_prompt_template(agent_type)

                    chain = LLMChain(
                        llm=llm,
                        prompt=prompt,
                        memory=memory,
                        verbose=True
                    )

                    def local_agent_function(state: NovelSystemState) -> Dict:
                        """The agent function to be used in the graph locally.

                        Args:
                            state: The current state.

                        Returns:
                            The updated state.
                        """
                        try:
                            # Prepare the context
                            context = {
                                "project_state": json.dumps(state["project"], indent=2),
                                "current_phase": state["project"].current_phase,
                                "task": state["current_input"].get("task", ""),
                                "input": state["current_input"].get("content", "")
                            }

                            # Run the chain
                            response = chain.run(input=json.dumps(context))

                            # Update the state
                            state["current_output"] = {
                                "agent": agent_type,
                                "content": response,
                                "timestamp": current_timestamp()
                            }

                            # Add to message history
                            state["messages"].append({
                                "role": agent_type,
                                "content": response,
                                "timestamp": current_timestamp()
                            })

                            return state
                        except Exception as e:
                            logger.error(f"Error in local agent function for agent {agent_type}: {e}")
                            state["errors"].append({
                                "agent": agent_type,
                                "error": str(e),
                                "timestamp": current_timestamp()
                            })
                            return state

                    return local_agent_function
            except Exception as e:
                logger.error(f"Error creating agent {agent_type}: {e}")
                raise
                
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
    
    def _get_llm(self, agent_type: str):
        """Retrieve the LLM for the given agent type."""
        config = MODEL_CONFIGS.get(agent_type, {})
        model_name = config.get("model", "anthropic/claude-3-opus-20240229")
        return self.assistants_client.get_llm(model_name)
    
    def _get_memory(self, agent_type: str, project_id: str):
        """Retrieve the memory for the given agent type."""
        return self.mongodb_manager.load_state(project_id)
    
    def _get_prompt_template(self, agent_type: str):
        """Retrieve the prompt template for the given agent type."""
        config = MODEL_CONFIGS.get(agent_type, {})
        return config.get("prompt_template", "")
    
    def create_agent(self, agent_type: str, state_type: Type[NovelSystemState], project_id: str) -> Callable:
        """Create an agent with the new agent_function."""
        tools = get_tools_for_agent(agent_type)
        return self.create_workflow_agent(agent_type, state_type, tools)
