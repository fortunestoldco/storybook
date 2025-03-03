import os
from typing import Dict, List, Optional, Any, Callable, Union
import json
import logging

# Chat models
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_aws.chat_models import ChatBedrock
from langchain_ollama import ChatOllama

# Core components
from langchain_core.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate
)
from langchain_core.memory import ConversationBufferMemory
from langchain_core.agents import AgentExecutor, ConversationalAgent, Tool
from langchain_core.tools import BaseTool
from langchain_core.chains import LLMChain
from langchain_core.messages import BaseMessage, SystemMessage

# Database
from langchain_mongodb import MongoDBChatMessageHistory

from langgraph.graph import StateGraph
from langgraph.graph.state import State

from langgraph.graph import Graph, StateGraph, END
from langgraph_sdk.client import SyncAssistantsClient

from config import MODEL_CONFIGS, PROMPT_TEMPLATES, MONGODB_CONFIG, OLLAMA_CONFIG
from prompts import get_prompt_for_agent
from state import NovelSystemState
from mongodb import MongoDBManager
from utils import create_prompt_with_context, current_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating agents in the novel writing system."""

    def __init__(self, mongo_manager: Optional[MongoDBManager] = None):
        """Initialize the agent factory.

        Args:
            mongo_manager: The MongoDB manager for persistence.
        """
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.assistants_client = SyncAssistantsClient()

    def _get_llm(self, agent_name: str) -> Any:
        """Get an LLM for an agent based on its configuration."""
        try:
            config = MODEL_CONFIGS.get(agent_name, {})
            model_name = config.get("model", "")
            
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
        except Exception as e:
            logger.error(f"Error getting LLM for agent {agent_name}: {e}")
            raise

    def _get_message_history(self, agent_name: str, project_id: str) -> MongoDBChatMessageHistory:
        """Get message history for an agent."""
        return MongoDBChatMessageHistory(
            connection_string=MONGODB_CONFIG["connection_string"],
            database_name=MONGODB_CONFIG["database_name"],
            collection_name=f"message_history_{agent_name}_{project_id}"
        )

    def _get_memory(self, agent_name: str, project_id: str) -> ConversationBufferMemory:
        """Get memory for an agent."""
        message_history = self._get_message_history(agent_name, project_id)
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )

    def _get_or_create_assistant(self, agent_name: str) -> str:
        """Get or create an assistant with LangGraph Cloud.

        Args:
            agent_name: Name of the agent.

        Returns:
            The assistant ID.
        """
        try:
            # Get the model configuration
            config = MODEL_CONFIGS.get(agent_name, {})
            model_name = config.get("model", "")
            
            # Get the specialized prompt for this agent
            prompt = get_prompt_for_agent(agent_name)
            
            # Create the assistant using LangGraph SDK
            assistant = self.assistants_client.create(
                name=f"{agent_name}_assistant",
                model=model_name,
                instructions=prompt,
                temperature=config.get("temperature", 0.3),
                tool_resources=[],
                tools=[]
            )
            return assistant.id
        except Exception as e:
            logger.error(f"Error creating assistant for agent {agent_name}: {e}")
            raise

    def create_agent(self, agent_name: str, project_id: str) -> Callable[[NovelSystemState], Dict]:
        """Create an agent function for use in the graph.

        Args:
            agent_name: Name of the agent.
            project_id: ID of the project.

        Returns:
            A callable agent function that takes NovelSystemState and returns Dict.
        """
        try:
            # For cloud deployment, use Assistants API
            if os.getenv("LANGGRAPH_CLOUD") >= "true":
                assistant_id = self._get_or_create_assistant(agent_name)
                
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
                            "agent": agent_name,
                            "content": response,
                            "timestamp": current_timestamp()
                        }

                        # Add to message history
                        state["messages"].append({
                            "role": agent_name,
                            "content": response,
                            "timestamp": current_timestamp()
                        })

                        return state
                    except Exception as e:
                        logger.error(f"Error in cloud agent function for agent {agent_name}: {e}")
                        state["errors"].append({
                            "agent": agent_name,
                            "error": str(e),
                            "timestamp": current_timestamp()
                        })
                        return state
                
                return cloud_agent_function
            
            # For local deployment, use LangChain
            else:
                llm = self._get_llm(agent_name)
                memory = self._get_memory(agent_name, project_id)
                prompt = self._get_prompt_template(agent_name)

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
                            "agent": agent_name,
                            "content": response,
                            "timestamp": current_timestamp()
                        }

                        # Add to message history
                        state["messages"].append({
                            "role": agent_name,
                            "content": response,
                            "timestamp": current_timestamp()
                        })

                        return state
                    except Exception as e:
                        logger.error(f"Error in local agent function for agent {agent_name}: {e}")
                        state["errors"].append({
                            "agent": agent_name,
                            "error": str(e),
                            "timestamp": current_timestamp()
                        })
                        return state

                return local_agent_function
        except Exception as e:
            logger.error(f"Error creating agent {agent_name}: {e}")
            raise

    def _get_prompt_template(self, agent_name: str) -> PromptTemplate:
        """Get a prompt template for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            A PromptTemplate instance.
        """
        try:
            # First try to get from the specialized prompts file
            template = get_prompt_for_agent(agent_name)
            
            # If not found, fall back to the basic templates in config
            if not template:
                template = PROMPT_TEMPLATES.get(agent_name, "You are an AI assistant.")

            # Create a ChatPromptTemplate
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
            chat_prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt
            ])

            return chat_prompt
        except Exception as e:
            logger.error(f"Error getting prompt template for agent {agent_name}: {e}")
            raise

# Replace ToolExecutor with our own implementation
class ToolExecutor:
    """Simple tool executor implementation"""
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
    
    def invoke(self, tool_name: str, tool_input: Any) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return self.tools[tool_name].run(tool_input)

class AgentState:
    """Represents the state of an agent in the workflow."""
    def __init__(self, **kwargs):
        self.data = kwargs
    
    def __getitem__(self, key):
        return self.data.get(key)
    
    def __setitem__(self, key, value):
        self.data[key] = value
