"""Base agent class for the storybook system."""

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate

from storybook.configuration import Configuration
from storybook.state import NovelSystemState
from storybook.prompts import get_agent_prompt


@dataclass
class AgentResult:
    """Result of an agent's processing."""
    message: AIMessage
    context: Dict[str, Any] = None


class BaseAgent:
    """Base class for all storybook agents."""
    
    def __init__(
        self,
        name: str,
        chat_model: BaseChatModel,
        tools: List[BaseTool],
        project_id: str,
        role_description: str
    ):
        """Initialize a base agent.
        
        Args:
            name: Agent name.
            chat_model: Language model to use.
            tools: List of tools available to the agent.
            project_id: ID of the project.
            role_description: Description of the agent's role.
        """
        self.name = name
        self.chat_model = chat_model
        self.tools = tools
        self.project_id = project_id
        self.role_description = role_description
        self.system_prompt_template = get_agent_prompt(name)
    
    def _build_system_prompt(self, state: NovelSystemState, config: Configuration) -> str:
        """Build the system prompt for the agent.
        
        Args:
            state: Current system state.
            config: System configuration.
            
        Returns:
            Formatted system prompt.
        """
        # Start with the base system prompt
        base_prompt = f"{config.system_prompt}\n\nYou are the {self.name}. {self.role_description}"
        
        # Add project and phase context
        context = f"\nProject ID: {self.project_id}\nCurrent Phase: {state.phase}\nTask: {state.current_input.get('task', '')}"
        
        # Format using the prompt template if available
        if self.system_prompt_template:
            format_dict = {
                "base_prompt": base_prompt,
                "context": context,
                "project_title": state.project.title,
                "genre": ', '.join(state.project.genre),
                "target_audience": ', '.join(state.project.target_audience),
                "phase": state.phase,
                "task": state.current_input.get('task', ''),
                "project_id": self.project_id,
            }
            return self.system_prompt_template.format(**format_dict)
        
        # Fallback to simple concatenation
        return f"{base_prompt}{context}"
    
    def _prepare_project_context(self, state: NovelSystemState) -> str:
        """Prepare context information about the project.
        
        Args:
            state: Current system state.
            
        Returns:
            Formatted project context.
        """
        return (
            f"Project title: {state.project.title}\n"
            f"Genre: {', '.join(state.project.genre)}\n"
            f"Target audience: {', '.join(state.project.target_audience)}\n"
            f"Current phase: {state.phase}"
        )
    
    async def process(self, state: NovelSystemState, config: Configuration) -> AgentResult:
        """Process the current state and return an agent result.
        
        Args:
            state: Current system state.
            config: System configuration.
            
        Returns:
            Agent result with the response message.
        """
        # Build system prompt
        system_prompt = self._build_system_prompt(state, config)
        
        # Prepare project context
        project_context = self._prepare_project_context(state)
        
        # Prepare messages for the model
        messages = [
            SystemMessage(content=system_prompt),
            *state.messages,
            AIMessage(content=f"Project context:\n{project_context}\n\nPlease address the current task: {state.current_input.get('task', '')}")
        ]
        
        # Get the model's response
        response = await self.chat_model.ainvoke(messages)
        
        # Return the result
        return AgentResult(message=response)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
