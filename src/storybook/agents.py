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
