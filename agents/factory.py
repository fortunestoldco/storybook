from typing import Dict, Any, Type, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from mongodb import MongoDBManager
from config import MODEL_CONFIGS, OLLAMA_CONFIG
from state import NovelSystemState
from tools.development import get_tools_for_agent
from prompts import get_prompt_for_agent
from utils import current_timestamp

import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating and managing agents in the LangGraph workflow."""

    def __init__(self, mongodb_manager: MongoDBManager):
        self.mongodb_manager = mongodb_manager
        self._agent_cache: Dict[str, Any] = {}
        self.phase_agents = {
            "initialization": ["quality_assessment_director", "plot_development_specialist", "character_psychology_specialist"],
            "development": ["creative_director", "structure_architect", "world_building_expert"],
            "creation": ["content_development_director", "chapter_drafters", "continuity_manager"],
            "refinement": ["editorial_director", "prose_enhancement_specialist", "grammar_consistency_checker"],
            "finalization": ["market_alignment_director", "hook_optimization_expert", "quality_assessment_director"]
        }

    def get_phase_agents(self, phase: str) -> list[str]:
        """Get all agent types for a specific phase."""
        return self.phase_agents.get(phase, [])

    def create_workflow_agent(
        self,
        agent_type: str,
        tools: Optional[list[BaseTool]] = None
    ) -> Callable:
        """Create a workflow-compatible agent function.

        Args:
            agent_type: Type of agent to create
            tools: Optional list of tools for the agent

        Returns:
            A callable agent function for use in the workflow
        """
        def agent_function(state: Dict) -> Dict:
            try:
                # Prepare the context
                context = {
                    "project_state": json.dumps(state["project"], indent=2),
                    "current_phase": state["current_phase"],
                    "task": state["current_input"].get("task", ""),
                    "input": state["current_input"].get("content", "")
                }

                # Get the LLM
                llm = self._get_llm(agent_type)
                
                # Get the prompt
                prompt_template = get_prompt_for_agent(agent_type)
                
                # Create a system and human message template
                system_message = SystemMessagePromptTemplate.from_template(prompt_template)
                human_message = HumanMessagePromptTemplate.from_template("{input}")
                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                
                # Create the chain
                chain = chat_prompt | llm
                
                # Run the chain
                response = chain.invoke({"input": json.dumps(context)})
                
                # Extract response text
                response_text = response.content if hasattr(response, 'content') else str(response)

                # Update the state
                state["current_output"] = {
                    "agent": agent_type,
                    "content": response_text,
                    "timestamp": current_timestamp()
                }

                # Add to message history
                state["messages"].append({
                    "role": agent_type,
                    "content": response_text,
                    "timestamp": current_timestamp()
                })

                return state
            except Exception as e:
                logger.error(f"Error in agent function for agent {agent_type}: {e}")
                state["errors"].append({
                    "agent": agent_type,
                    "error": str(e),
                    "timestamp": current_timestamp()
                })
                return state

        return agent_function

    def _get_llm(self, agent_type: str) -> Any:
        """Get the LLM for the given agent type."""
        config = MODEL_CONFIGS.get(agent_type, {})
        model_name = config.get("model", "anthropic/claude-3-opus-20240229")
        
        if model_name.startswith("anthropic/"):
            return ChatAnthropic(
                model=model_name.split("/")[1],
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4000),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif model_name.startswith("openai/"):
            return ChatOpenAI(
                model=model_name.split("/")[1],
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 4000)
            )
        elif model_name.startswith("ollama/"):
            return ChatOllama(
                model=model_name.split("/")[1],
                temperature=config.get("temperature", 0.2),
                base_url=OLLAMA_CONFIG["host"]
            )
        else:
            raise ValueError(f"Unsupported model provider for {model_name}")

    def create_agent(self, agent_type: str, state_type: Type, project_id: str) -> Callable:
        """Create an agent with tools."""
        tools = get_tools_for_agent(agent_type)
        return self.create_workflow_agent(agent_type, tools)
