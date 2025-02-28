from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

class BaseAgent(ABC):
    """Base class for all storybook agents."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config or {}
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        llm_type = self.llm_config.get("type", "gpt-4")
        
        if llm_type.startswith("gpt"):
            return ChatOpenAI(
                model_name=llm_type,
                temperature=self.llm_config.get("temperature", 0.7)
            )
        elif llm_type.startswith("claude"):
            return ChatAnthropic(
                model=llm_type,
                temperature=self.llm_config.get("temperature", 0.7)
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def generate_content(self, system_prompt: str, human_prompt: str) -> str:
        """Generate content using the configured LLM."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = self.llm.generate([messages])
        return response.generations[0][0].text

    @abstractmethod
    def process_manuscript(self, manuscript_id: str) -> Dict[str, Any]:
        """Process manuscript with agent-specific logic."""
        pass