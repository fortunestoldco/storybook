from typing import Dict, List, Optional, Any, Callable
import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_aws import BedrockChat
from langchain_google_vertexai import ChatVertexAI
from langchain_azure_openai import AzureChatOpenAI
from langchain_community.llms import Ollama, LlamaCpp, Replicate
from langchain_community.chat_models import ChatOllama
from langchain_mongodb import MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ConversationalAgent
from langchain.chains import LLMChain

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongo import MongoDBCheckpointHandler

from config import MODEL_CONFIGS, PROMPT_TEMPLATES, MONGODB_CONFIG
from backend import BackendProvider, BackendConfig, get_default_backend_config
from state import NovelSystemState
from mongodb import MongoDBManager
from utils import create_prompt_with_context, current_timestamp
from prompts import get_prompt_for_agent


class AgentFactory:
    """Factory for creating agents in the storybook system."""
    
    def __init__(self, mongo_manager: Optional[MongoDBManager] = None, backend_config: Optional[BackendConfig] = None):
        """Initialize the agent factory.
        
        Args:
            mongo_manager: The MongoDB manager for persistence.
            backend_config: The backend configuration for LLM providers.
        """
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.backend_config = backend_config or get_default_backend_config()
    
    def _get_llm(self, agent_name: str) -> Any:
        """Get an LLM for an agent based on its configuration.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            An LLM instance.
        """
        config = MODEL_CONFIGS.get(agent_name, {})
        model_name = config.get("model", "")
        
        # Get the specific endpoint for this model if available
        model_endpoint = None
        if self.backend_config.model_endpoints:
            for model_key, endpoint in self.backend_config.model_endpoints.items():
                if model_key in model_name:
                    model_endpoint = endpoint
                    break
        
        # Process based on the backend provider
        if self.backend_config.provider == BackendProvider.AWS_BEDROCK:
            # Clean model name to work with Bedrock
            if model_name.startswith("anthropic/"):
                clean_model_name = model_name.replace("anthropic/", "")
            elif model_name.startswith("meta-llama/") or model_name.startswith("llama-3/"):
                clean_model_name = model_name.replace("meta-llama/", "").replace("llama-3/", "")
            elif model_name.startswith("google/"):
                clean_model_name = model_name.replace("google/", "")
            elif model_name.startswith("mistralai/"):
                clean_model_name = model_name.replace("mistralai/", "")
            elif model_name.startswith("microsoft/"):
                clean_model_name = model_name.replace("microsoft/", "")
            elif model_name.startswith("databricks/"):
                clean_model_name = model_name.replace("databricks/", "")
            elif model_name.startswith("Qwen/"):
                clean_model_name = model_name.replace("Qwen/", "")
            elif model_name.startswith("NousResearch/"):
                clean_model_name = model_name.replace("NousResearch/", "")
            elif model_name.startswith("Salesforce/"):
                clean_model_name = model_name.replace("Salesforce/", "")
            else:
                clean_model_name = model_name
            
            return BedrockChat(
                model_id=clean_model_name,
                model_kwargs={
                    "temperature": config.get("temperature", 0.3),
                    "max_tokens": config.get("max_tokens", 2000)
                },
                region_name=self.backend_config.region
            )
        
        elif self.backend_config.provider == BackendProvider.HUGGINGFACE:
            # For HuggingFace, we need an endpoint URL
            if not model_endpoint and not self.backend_config.api_url:
                raise ValueError(f"HuggingFace endpoint URL required for model {model_name}")
            
            from langchain_community.llms import HuggingFaceEndpoint
            
            return HuggingFaceEndpoint(
                endpoint_url=model_endpoint or self.backend_config.api_url,
                huggingfacehub_api_token=self.backend_config.api_key,
                task="text-generation",
                model_kwargs={
                    "temperature": config.get("temperature", 0.3),
                    "max_tokens": config.get("max_tokens", 2000)
                }
            )
        
        elif self.backend_config.provider == BackendProvider.AZURE_OPENAI:
            # For Azure, we need deployment name
            if not self.backend_config.deployment_name:
                raise ValueError(f"Azure OpenAI deployment name required for model {model_name}")
            
            return AzureChatOpenAI(
                deployment_name=self.backend_config.deployment_name,
                openai_api_key=self.backend_config.api_key,
                azure_endpoint=self.backend_config.api_url,
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 2000)
            )
        
        elif self.backend_config.provider == BackendProvider.GOOGLE_VERTEX:
            if not self.backend_config.project_id:
                raise ValueError(f"Google project ID required for model {model_name}")
            
            # Clean model name to work with Vertex AI
            if model_name.startswith("google/"):
                clean_model_name = model_name.replace("google/", "")
            else:
                clean_model_name = model_name
            
            return ChatVertexAI(
                model_name=clean_model_name,
                project=self.backend_config.project_id,
                temperature=config.get("temperature", 0.3),
                max_output_tokens=config.get("max_tokens", 2000)
            )
        
        elif self.backend_config.provider == BackendProvider.OLLAMA:
            # For Ollama, we need a URL
            if not self.backend_config.api_url:
                raise ValueError(f"Ollama API URL required for model {model_name}")
            
            # Extract base model name without provider prefix
            if "/" in model_name:
                _, base_model = model_name.split("/", 1)
            else:
                base_model = model_name
            
            return ChatOllama(
                model=base_model,
                base_url=self.backend_config.api_url,
                temperature=config.get("temperature", 0.3)
            )
        
        elif self.backend_config.provider == BackendProvider.LLAMACPP:
            # For llama.cpp, we need a model path or server URL
            if not self.backend_config.api_url:
                raise ValueError(f"llama.cpp model path or server URL required for model {model_name}")
            
            return LlamaCpp(
                model_path=self.backend_config.api_url,
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 2000),
                n_gpu_layers=-1,  # Automatically use as many layers as possible on GPU
                n_ctx=4096         # Context window size
            )
        
        elif self.backend_config.provider == BackendProvider.REPLICATE:
            # For Replicate, we need an API key
            if not self.backend_config.api_key:
                raise ValueError(f"Replicate API key required for model {model_name}")
            
            # Model format for Replicate is typically: username/model_name:version
            return Replicate(
                model=model_name,
                api_key=self.backend_config.api_key,
                input={
                    "temperature": config.get("temperature", 0.3),
                    "max_new_tokens": config.get("max_tokens", 2000)
                }
            )
        
        elif self.backend_config.provider == BackendProvider.CUSTOM:
            # For custom backend, use the provided endpoint
            if not model_endpoint and not self.backend_config.api_url:
                raise ValueError(f"Custom endpoint URL required for model {model_name}")
            
            # For custom, user needs to specify the appropriate LangChain class in their code
            # This is just a placeholder implementation that uses OpenAI-compatible API
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=self.backend_config.api_key,
                openai_api_base=model_endpoint or self.backend_config.api_url,
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 2000)
            )
        
        else:
            # Default to AnthropicAI for Claude models, otherwise OpenAI
            if model_name.startswith("anthropic/") or "claude" in model_name.lower():
                clean_model_name = model_name.replace("anthropic/", "")
                return ChatAnthropic(
                    model_name=clean_model_name,
                    temperature=config.get("temperature", 0.3),
                    max_tokens=config.get("max_tokens", 2000)
                )
            else:
                return ChatOpenAI(
                    model_name=model_name,
                    temperature=config.get("temperature", 0.3),
                    max_tokens=config.get("max_tokens", 2000)
                )
    
    def _get_message_history(self, agent_name: str, project_id: str) -> MongoDBChatMessageHistory:
        """Get message history for an agent from MongoDB.
        
        Args:
            agent_name: Name of the agent.
            project_id: ID of the project.
            
        Returns:
            A MongoDBChatMessageHistory instance.
        """
        return MongoDBChatMessageHistory(
            connection_string=MONGODB_CONFIG["connection_string"],
            database_name=MONGODB_CONFIG["database_name"],
            collection_name=f"message_history_{agent_name}",
            session_id=project_id
        )
    
    def _get_memory(self, agent_name: str, project_id: str) -> ConversationBufferMemory:
        """Get memory for an agent.
        
        Args:
            agent_name: Name of the agent.
            project_id: ID of the project.
            
        Returns:
            A ConversationBufferMemory instance.
        """
        message_history = self._get_message_history(agent_name, project_id)
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )
    
    def _get_prompt_template(self, agent_name: str) -> PromptTemplate:
        """Get a prompt template for an agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            A PromptTemplate instance.
        """
        # Use extended prompts from prompts.py if available
        template = get_prompt_for_agent(agent_name) or PROMPT_TEMPLATES.get(agent_name, "You are an AI assistant.")
        
        # Create a ChatPromptTemplate
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        return chat_prompt
    
    def create_agent(self, agent_name: str, project_id: str) -> Callable:
        """Create an agent function for use in the graph.
        
        Args:
            agent_name: Name of the agent.
            project_id: ID of the project.
            
        Returns:
            A callable agent function.
        """
        llm = self._get_llm(agent_name)
        memory = self._get_memory(agent_name, project_id)
        prompt = self._get_prompt_template(agent_name)
        
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        def agent_function(state: NovelSystemState) -> Dict:
            """The agent function to be used in the graph.
            
            Args:
                state: The current state.
                
            Returns:
                The updated state.
            """
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
            
            # Add to messages
            state["messages"].append({
                "role": agent_name,
                "content": response,
                "timestamp": current_timestamp()
            })
            
            return state
        
        return agent_function
