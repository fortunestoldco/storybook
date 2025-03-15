import os
import warnings
import traceback
from typing import Dict, Any, List, Optional
import torch
import queue
import time
import functools
from datetime import datetime
import re
import threading

from pymongo import MongoClient
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tools import ResearchTools
from utils import retry_with_exponential_backoff, cleanup_memory, extract_chunk_references
from prompts import DEFAULT_PROMPTS
from config import HUGGINGFACE_API_TOKEN, MONGODB_URI, LANGSMITH_API_KEY, LANGSMITH_TRACING_ENABLED, LANGSMITH_PROJECT

# Utility functions for manuscript updates
def is_content_creation_agent(agent_name: str) -> bool:
    """Determine if the agent is a content creation/refinement agent that produces rewrites."""
    content_creation_agents = {
        "chapter_drafters",
        "scene_construction_specialists",
        "dialogue_crafters",
        "prose_enhancement_specialist",
        "dialogue_refinement_expert"
    }
    return agent_name in content_creation_agents

def extract_rewritten_content(response_content: str) -> str:
    """
    Extract rewritten content from an agent's response.
    Looks for content that appears to be a rewritten manuscript section.
    """
    # First, try to find content between code block markers
    code_block_pattern = r'```(?:[\w]*\n)?(.*?)```'
    code_block_matches = re.findall(code_block_pattern, response_content, re.DOTALL)
    if code_block_matches:
        # Take the longest code block as it's likely the rewritten content
        return max(code_block_matches, key=len)

    # Look for content after headings like "Rewritten Content:" or "Improved Version:"
    heading_patterns = [
        r'(?:rewritten|improved|revised|enhanced)(?:\s+)(?:content|version|text|section|chapter|dialogue|scene|passage)(?:\s*):(?:\s*)(.*)',
        r'(?:here\s+is\s+the\s+|i\s+have\s+|the\s+)(?:rewritten|improved|revised|enhanced)(?:\s+)(?:content|version|text|section|chapter|dialogue|scene|passage)(?:\s*):(?:\s*)(.*)',
        r'(?:my\s+rewrite|my\s+revision|my\s+improvement)(?:\s*):(?:\s*)(.*)'
    ]

    for pattern in heading_patterns:
        matches = re.findall(pattern, response_content.lower(), re.DOTALL)
        if matches:
            # Take the longest match
            return max(matches, key=len)

    # If we can't find explicit markers, try to identify paragraphs that look like narrative content
    paragraphs = response_content.split('\n\n')
    narrative_paragraphs = []

    for paragraph in paragraphs:
        # Ignore short paragraphs or those that look like commentary
        if len(paragraph.strip()) < 50:
            continue
        if re.search(r'^(?:I |As the |Here |To |This |The |In |For )', paragraph):
            continue
        narrative_paragraphs.append(paragraph)

    if narrative_paragraphs:
        return '\n\n'.join(narrative_paragraphs)

    # If all else fails, look for the longest paragraph that might be content
    if paragraphs:
        # Find the longest paragraph that's not obviously commentary
        content_paragraphs = [p for p in paragraphs if len(p.strip()) > 100]
        if content_paragraphs:
            return max(content_paragraphs, key=len)

    # If all extraction methods fail, return an empty string
    return ""

def update_manuscript_with_rewrite(state, agent_name, response_content):
    """
    Update manuscript chunks with rewritten content from a content creation agent.
    Returns the updated state.
    """
    # Only process content creation agents
    if not is_content_creation_agent(agent_name):
        return state

    # Extract rewritten content
    rewritten_content = extract_rewritten_content(response_content)
    if not rewritten_content or len(rewritten_content.strip()) < 50:  # Ignore very short content
        return state

    # Get referenced chunks from current_input
    referenced_chunks = state.get("current_input", {}).get("referenced_chunks", [])

    # If no specific chunks are referenced, check if any sections are specified
    if not referenced_chunks:
        section_to_review = state.get("current_input", {}).get("section_to_review", "")
        chapter_to_review = state.get("current_input", {}).get("chapter_to_review", "")

        # If we have a section or chapter to review, try to find matching chunks
        if section_to_review or chapter_to_review:
            search_term = section_to_review or chapter_to_review
            search_term_lower = search_term.lower()

            for i, chunk in enumerate(state.get("project", {}).get("manuscript_chunks", [])):
                if search_term_lower in chunk.get("content", "").lower():
                    referenced_chunks.append(i)

    # If we still don't have any referenced chunks, use manuscript_excerpt to find matching chunks
    if not referenced_chunks:
        manuscript_excerpt = ""
        for msg in state.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                excerpt_matches = re.findall(r'Manuscript section to review:(.*?)(?:Task:|Context:|$)', content, re.DOTALL)
                if excerpt_matches:
                    manuscript_excerpt = excerpt_matches[0].strip()
                    break

        if manuscript_excerpt and len(manuscript_excerpt) > 50:
            for i, chunk in enumerate(state.get("project", {}).get("manuscript_chunks", [])):
                if manuscript_excerpt[:50] in chunk.get("content", ""):
                    referenced_chunks.append(i)
                    break

    # If we still don't have any referenced chunks, use the first chunk as default
    if not referenced_chunks and state.get("project", {}).get("manuscript_chunks"):
        referenced_chunks = [0]  # Default to first chunk

    # Update the referenced chunks with the rewritten content
    if referenced_chunks and state.get("project", {}).get("manuscript_chunks"):
        project = state.get("project", {}).copy()
        manuscript_chunks = project.get("manuscript_chunks", []).copy()

        # If multiple chunks are referenced, try to split the rewritten content accordingly
        if len(referenced_chunks) > 1:
            # Simple approach: divide content roughly by the number of chunks
            content_per_chunk = len(rewritten_content) // len(referenced_chunks)
            for i, chunk_idx in enumerate(referenced_chunks):
                if chunk_idx < len(manuscript_chunks):
                    start_pos = i * content_per_chunk
                    end_pos = (i + 1) * content_per_chunk if i < len(referenced_chunks) - 1 else len(rewritten_content)
                    chunk_content = rewritten_content[start_pos:end_pos]
                    manuscript_chunks[chunk_idx] = manuscript_chunks[chunk_idx].copy()  # Create a copy to avoid mutating the original
                    manuscript_chunks[chunk_idx]["content"] = chunk_content
        else:
            # Single chunk update
            chunk_idx = referenced_chunks[0]
            if chunk_idx < len(manuscript_chunks):
                manuscript_chunks[chunk_idx] = manuscript_chunks[chunk_idx].copy()  # Create a copy to avoid mutating the original
                manuscript_chunks[chunk_idx]["content"] = rewritten_content

        # Update the project with the modified chunks
        project["manuscript_chunks"] = manuscript_chunks

        # Reassemble the full manuscript
        full_manuscript = reassemble_manuscript(manuscript_chunks)
        project["manuscript"] = full_manuscript

        # Update the state with the modified project
        updated_state = state.copy()
        updated_state["project"] = project
        return updated_state

    return state

def reassemble_manuscript(manuscript_chunks):
    """
    Reassemble the full manuscript from chunks.
    """
    # Sort chunks by chunk_id to ensure correct order
    sorted_chunks = sorted(manuscript_chunks, key=lambda x: x.get("chunk_id", 0))

    # Combine the content from all chunks
    full_manuscript = ""
    for chunk in sorted_chunks:
        content = chunk.get("content", "")
        # Add spacing between chunks if needed
        if full_manuscript and not (full_manuscript.endswith("\n") or content.startswith("\n")):
            full_manuscript += "\n\n"
        full_manuscript += content

    return full_manuscript

# Setup LangSmith tracer
langchain_tracer = None
langsmith_project_name = LANGSMITH_PROJECT
if LANGSMITH_TRACING_ENABLED and LANGSMITH_API_KEY:
    langchain_tracer = LangChainTracer(
        project_name=langsmith_project_name,
    )
    # Set debug to get more detailed tracing
    from langchain_core.globals import set_debug
    set_debug(True)
    print(f"LangSmith tracing enabled for project: {langsmith_project_name}")
else:
    print("LangSmith tracing not enabled. Check LANGSMITH_API_KEY and LANGSMITH_TRACING environment variables.")

# Initialize the callback manager for streaming
streaming_callback = StreamingStdOutCallbackHandler()
callback_manager = CallbackManager([streaming_callback])
if langchain_tracer:
    callback_manager.add_handler(langchain_tracer)

class AgentFactory:
    def __init__(self, config, model_config=None, tavily_client=None):
        # Check for HuggingFace token
        self.huggingface_token = HUGGINGFACE_API_TOKEN
        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

        # Log into Hugging Face
        login(token=self.huggingface_token)

        # Initialize with default model config if none provided
        self.default_model_config = model_config or {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text-generation",
            "temperature": 0.1,
            "max_new_tokens": 512,
            "do_sample": True,  # Changed from False to True
            "repetition_penalty": 1.03
        }

        # Initialize agent-specific model configs
        self.agent_model_configs = {}
        if isinstance(model_config, dict) and model_config.get("agent_configs"):
            self.agent_model_configs = model_config.get("agent_configs")

        # Setup callback manager for streaming and tracing BEFORE creating the model
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        if langchain_tracer:
            self.callback_manager.add_handler(langchain_tracer)

        self.config = config
        self.tavily = tavily_client or ResearchTools()

        # Initialize models dictionary to store agent-specific models
        self.models = {}

        # Add model cache with LRU policy
        self.model_cache = {}
        self.model_cache_max_size = 3  # Maximum number of models to keep in cache
        self.model_usage_tracker = []  # Track model usage for LRU policy

        # MongoDB connection for prompts
        self.mongo_client = None
        mongo_uri = MONGODB_URI
        if mongo_uri:
            try:
                self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                self.mongo_client.admin.command('ping')  # Test connection
                print("MongoDB connection for AgentFactory successful")
            except Exception as e:
                print(f"Warning: Could not connect to MongoDB: {str(e)}. Using default prompts.")

    def load_prompt_from_mongodb(self, agent_name):
        """Load agent prompt from MongoDB, returning None if not found"""
        if not self.mongo_client:
            return None

        try:
            db = self.mongo_client["storybook"]
            prompts_collection = db["prompts"]

            # Try to find the agent's prompt
            doc = prompts_collection.find_one({"agent_name": agent_name})
            if doc and "prompt_text" in doc:
                print(f"Loaded prompt for {agent_name} from MongoDB")
                return doc["prompt_text"]
        except Exception as e:
            print(f"Error loading prompt from MongoDB for {agent_name}: {str(e)}")

        return None

    def _create_huggingface_model(self, model_config):
        """Create a HuggingFacePipeline model with chat template support for multi-GPU inference."""
        try:
            # Import necessary libs
            from transformers import AutoTokenizer, pipeline

            # Check for CUDA availability more carefully
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device type: {device_type}")

            # Set torch dtype based on device and availability
            if device_type == "cuda":
                # Check if bfloat16 is supported, otherwise fall back to float16
                torch_dtype = torch.bfloat16 if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float16
            else:
                # For CPU, use bfloat16 if supported by CPU architecture, otherwise float32
                torch_dtype = torch.bfloat16 if hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported() else torch.float32

            print(f"Using torch dtype: {torch_dtype}")

            # Get model ID from config
            model_id = model_config.get("model_id", "HuggingFaceH4/zephyr-7b-beta")
            print(f"Loading model: {model_id}")

            # Model kwargs with safer settings to prevent device errors
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch_dtype,
            }

            # Only set device_map if safe to do so
            if device_type == "cuda":
                # For GPU, use more explicit device mapping to avoid device errors
                if torch.cuda.device_count() == 1:
                    model_kwargs["device_map"] = 0  # Explicitly set to first GPU
                else:
                    # For multiple GPUs, use balanced mapping instead of auto
                    model_kwargs["device_map"] = "balanced"
            else:
                # For CPU, explicitly set to CPU
                model_kwargs["device_map"] = "cpu"

            # Load the tokenizer first to set up the chat template
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.huggingface_token
            )

            # Check if tokenizer has a chat template
            if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
                print(f"No chat template found for model {model_id}, adding a default template")

                # Set a default ChatML-style template which works for many models
                default_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}<|im_end|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_end|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_end|>assistant
{{ message['content'] }}<|im_end|>
{% else %}<|im_end|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_end|>assistant
{% endif %}"""

                tokenizer.chat_template = default_template
                print("Added default ChatML template")

            print("Creating pipeline with configured tokenizer")

            # Create the text generation pipeline
            text_gen_pipeline = pipeline(
                task=model_config.get("task", "text-generation"),
                model=model_id,
                tokenizer=tokenizer,
                max_new_tokens=model_config.get("max_new_tokens", 512),
                temperature=model_config.get("temperature", 0.1),
                do_sample=model_config.get("do_sample", True),
                repetition_penalty=model_config.get("repetition_penalty", 1.03),
                return_full_text=False,
                token=self.huggingface_token,
                model_kwargs=model_kwargs
            )

            # Get the underlying model and configure it safely
            if hasattr(text_gen_pipeline, "model"):
                model = text_gen_pipeline.model

                # Only set static KV cache if the attribute exists and we're on CUDA
                if device_type == "cuda" and hasattr(model, "generation_config") and \
                hasattr(model.generation_config, 'cache_implementation'):
                    try:
                        model.generation_config.cache_implementation = "static"
                        print("Static KV cache enabled for optimized inference")
                    except Exception as cache_err:
                        print(f"Could not set static KV cache: {cache_err}")

            # Create HuggingFacePipeline from the pipeline
            llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

            # Create ChatHuggingFace from the pipeline
            chat_model = ChatHuggingFace(llm=llm, callbacks=self.callback_manager)
            return chat_model

        except Exception as e:
            print(f"Error creating HuggingFace model: {str(e)}")
            traceback.print_exc()
            # Fallback to a smaller model if there's an issue
            try:
                print("Falling back to small model (google/flan-t5-base)...")

                # Load tokenizer with a chat template for the fallback model
                from transformers import AutoTokenizer, pipeline
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", token=self.huggingface_token)

                # Set a simple chat template for T5 models
                simple_template = """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% else %}{{ message['role'] }}: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""

                tokenizer.chat_template = simple_template

                # Create the text generation pipeline for fallback model
                text_gen_pipeline = pipeline(
                    task="text2text-generation",
                    model="google/flan-t5-base",
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.1,
                    model_kwargs={
                        "device_map": "auto",
                        "torch_dtype": torch.float32,
                    },
                    token=self.huggingface_token
                )

                # Create HuggingFacePipeline from the pipeline
                llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

                # Create ChatHuggingFace from the pipeline
                chat_model = ChatHuggingFace(llm=llm, callbacks=self.callback_manager)
                return chat_model
            except Exception as fallback_error:
                print(f"Fallback model also failed: {fallback_error}")
                traceback.print_exc()
                # Last resort - create a simple mock model that won't crash
                return SafeFallbackModel(agent_name_="fallback")

    def _update_model_cache(self, agent_name, model):
        """Update the model cache using LRU policy"""
        # If model already in cache, update usage
        if agent_name in self.model_cache:
            # Remove from usage tracker and add to end (most recently used)
            if agent_name in self.model_usage_tracker:
                self.model_usage_tracker.remove(agent_name)
            self.model_usage_tracker.append(agent_name)
            return

        # If cache is full, remove least recently used model
        if len(self.model_cache) >= self.model_cache_max_size:
            # Get the least recently used agent
            if self.model_usage_tracker:
                lru_agent = self.model_usage_tracker.pop(0)
                if lru_agent in self.model_cache:
                    print(f"Removing {lru_agent} model from cache (LRU policy)")
                    # Remove the model from cache
                    del self.model_cache[lru_agent]
                    # Force cleanup
                    cleanup_memory()

        # Add new model to cache
        self.model_cache[agent_name] = model
        self.model_usage_tracker.append(agent_name)

    def get_model(self, agent_name=None):
        """Lazy-load the model for the specified agent or the default model with efficient caching."""
        cache_key = agent_name or "default"

        # Check if model is in cache
        if cache_key in self.model_cache:
            print(f"Using cached model for {cache_key}")
            # Update usage tracking
            if cache_key in self.model_usage_tracker:
                self.model_usage_tracker.remove(cache_key)
            self.model_usage_tracker.append(cache_key)
            return self.model_cache[cache_key]

        # Not in cache, need to load model
        print(f"Model for {cache_key} not in cache, loading...")

        # Determine which model config to use
        model_config = self.default_model_config
        if agent_name and agent_name in self.agent_model_configs:
            model_config = self.agent_model_configs[agent_name]
            print(f"Using agent-specific model for {agent_name}: {model_config['model_id']}")

        try:
            print(f"Creating HuggingFace model for {agent_name or 'default'} with {model_config.get('model_id', 'unknown')}")
            model = self._create_huggingface_model(model_config)

            # Update model cache
            self._update_model_cache(cache_key, model)

            print(f"Model creation successful for {agent_name or 'default'}")
            return model
        except Exception as e:
            print(f"Error during model creation for {agent_name or 'default'}: {str(e)}")
            print("Creating safe fallback model...")

            # Create a fallback model
            fallback = SafeFallbackModel(agent_name_=agent_name or "default")
            # Don't cache fallback models
            return fallback

    def create_research_agent(self, research_type: str):
        """Create a research agent function."""

        def research_agent_function(state):
            """Research agent function that processes the current state."""
            project = state.get("project", {})
            current_input = state.get("current_input", {})

            # Extract research query from task or create one based on context
            research_query = current_input.get("research_query", "")
            if not research_query:
                task = current_input.get("task", "")
                manuscript_excerpt = ""
                if project.get("manuscript_chunks"):
                    manuscript_excerpt = project["manuscript_chunks"][0]["content"][:500]

                if research_type == "domain":
                    research_query = f"Technical information about: {task}"
                elif research_type == "cultural":
                    research_query = f"Cultural context related to: {task}"
                elif research_type == "market":
                    research_query = f"Market trends and audience preferences for: {task}"
                else:
                    research_query = f"Information about: {task}"

                # Add context from manuscript
                if manuscript_excerpt:
                    research_query += f" Context from manuscript: {manuscript_excerpt}"

            try:
                # Use the appropriate research function from the ResearchTools class
                if research_type == "domain":
                    search_result = self.tavily.domain_research(research_query)
                elif research_type == "cultural":
                    search_result = self.tavily.cultural_research(research_query)
                elif research_type == "market":
                    search_result = self.tavily.market_research(research_query)
                else:
                    # Default to domain research
                    search_result = self.tavily.domain_research(research_query)

                # Extract results and summary
                results = search_result.get("results", [])
                summary = search_result.get("summary", "No summary available")

                # Format research results
                research_results = f"RESEARCH TYPE: {research_type.upper()}\n\n"
                research_results += f"QUERY: {research_query}\n\n"
                research_results += f"SUMMARY: {summary}\n\n"
                research_results += "DETAILS:\n" + "\n\n".join(
                    [f"- {r['title']}: {r['content'][:300]}..." for r in results[:3]]
                )

                # Update state with research results
                updated_state = state.copy()
                updated_state["current_input"] = current_input.copy()
                updated_state["current_input"]["research_results"] = research_results
                updated_state["count"] = state.get("count", 0) + 1
                updated_state["lnode"] = f"{research_type}_research"

                # Add to messages for tracking
                updated_state["messages"] = state.get("messages", []) + [
                    {"role": "system", "content": f"Conducted {research_type} research on: {research_query}"},
                    {"role": "assistant", "content": research_results}
                ]

                return updated_state

            except Exception as e:
                print(f"Error conducting research: {str(e)}")

                # Update state with error
                updated_state = state.copy()
                updated_state["current_input"] = current_input.copy()
                updated_state["current_input"]["research_results"] = f"Research error: {str(e)}"
                updated_state["count"] = state.get("count", 0) + 1
                updated_state["lnode"] = f"{research_type}_research"

                # Add to messages for tracking
                updated_state["messages"] = state.get("messages", []) + [
                    {"role": "system", "content": f"Attempted {research_type} research on: {research_query}"},
                    {"role": "assistant", "content": f"Research error: {str(e)}"}
                ]

                return updated_state

        return research_agent_function

    def create_agent(self, agent_name: str, project_id: str):
        """Create a function for a specific agent."""
        # First, try to load prompt from MongoDB
        agent_prompt = None
        if self.mongo_client:
            agent_prompt = self.load_prompt_from_mongodb(agent_name)

        # If not found in MongoDB, use default prompts
        if not agent_prompt:
            if agent_name in DEFAULT_PROMPTS:
                agent_prompt = DEFAULT_PROMPTS[agent_name]
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        # Create a function that uses the agent's prompt to process inputs
        def agent_function(state):
            """Agent function that processes the current state."""
            project = state.get("project", {})
            current_input = state.get("current_input", {})

            # Get relevant manuscript content
            manuscript_excerpt = ""
            section_to_review = current_input.get("section_to_review", "")
            chapter_to_review = current_input.get("chapter_to_review", "")
            referenced_chunks = current_input.get("referenced_chunks", [])

            if project.get("manuscript_chunks"):
                # If we have a specific section or chapter to review, try to find it
                if section_to_review or chapter_to_review:
                    search_term = section_to_review or chapter_to_review
                    # Convert to lowercase for case-insensitive search
                    search_term_lower = search_term.lower()

                    # Search through chunks for the specified section
                    for chunk in project["manuscript_chunks"]:
                        if search_term_lower in chunk["content"].lower():
                            manuscript_excerpt = chunk["content"]
                            break

                    # If nothing was found but we're the executive director, include more context
                    if not manuscript_excerpt and agent_name == "executive_director":
                        # For executive director, provide a larger portion of the manuscript
                        manuscript_excerpt = "\n\n".join([
                            f"Chunk {chunk['chunk_id']}: {chunk['content']}"
                            for chunk in project["manuscript_chunks"][:10]  # Increased from 3 to 10 chunks
                        ])
                # If specific chunks were referenced in the delegation
                elif referenced_chunks:
                    manuscript_excerpt = "\n\n".join([
                        f"Chunk {chunk['chunk_id']}: {chunk['content']}"
                        for chunk in project["manuscript_chunks"]
                        if chunk["chunk_id"] in referenced_chunks
                    ])
                else:
                    # For executive director, provide comprehensive manuscript view
                    if agent_name == "executive_director":
                        # Provide a larger portion of the manuscript with chunk identifiers
                        manuscript_excerpt = "\n\n".join([
                            f"Chunk {chunk['chunk_id']}: {chunk['content']}"
                            for chunk in project["manuscript_chunks"][:10]  # Increased from 3 to 10 chunks
                        ])
                    # For other directors, provide moderate context
                    elif agent_name in ["creative_director", "editorial_director", "content_development_director", "market_alignment_director"]:
                        manuscript_excerpt = "\n\n".join([
                            f"Chunk {chunk['chunk_id']}: {chunk['content']}"
                            for chunk in project["manuscript_chunks"][:5]  # Moderate amount for other directors
                        ])
                    else:
                        # For specialists, just the first chunk as a sample if no specific chunks are referenced
                        manuscript_excerpt = f"Chunk 0: {project['manuscript_chunks'][0]['content']}"

            # Create the context string
            context = (
                f"Project ID: {project.get('id', project_id)}\n"
                f"Title: {project.get('title', 'Untitled')}\n"
                f"Synopsis: {project.get('synopsis', 'No synopsis provided')}\n"
                f"Current task: {current_input.get('task', 'No task specified')}\n"
            )

            # Add research context if available
            research_context = ""
            if "research_results" in current_input:
                research_context = f"\nResearch Results:\n{current_input['research_results']}\n"

            # Make sure we have valid content
            system_content = agent_prompt.strip()
            if not system_content:
                system_content = "You are an AI assistant helping with a writing project."

            # For executive director, add specific instruction to provide detailed feedback
            if agent_name == "executive_director":
                system_content += "\n\nIMPORTANT ADDITIONAL INSTRUCTIONS: You must thoroughly analyze the manuscript chunks provided below. For each chunk, identify specific issues, strengths, and provide detailed feedback. Then delegate SPECIFIC tasks to appropriate specialists. Always reference the exact chunks (e.g., 'Chunk 3 needs work on character development') when delegating tasks."

            # For content creators (chapter drafters, dialogue crafters, prose specialists), emphasize rewriting
            if agent_name in ["chapter_drafters", "dialogue_crafters", "prose_enhancement_specialist", "scene_construction_specialists", "dialogue_refinement_expert"]:
                system_content += "\n\nIMPORTANT ADDITIONAL INSTRUCTIONS: Your primary job is to REWRITE and IMPROVE the manuscript sections you're given. Don't just give feedback or suggestions - actually provide completely rewritten text that significantly improves on the original. Focus on transforming the text from draft quality to professional, published-quality writing."

            # For specialized agents, add context about previous director instructions
            if agent_name not in ["executive_director", "creative_director", "editorial_director", "content_development_director", "market_alignment_director"]:
                # Find the last message from a director
                director_instructions = []
                for msg in reversed(state.get("messages", [])):
                    if msg.get("role") == "assistant" and any(director in msg.get("content", "") for director in ["Executive Director", "Creative Director", "Editorial Director", "Content Development Director", "Market Alignment Director"]):
                        director_instructions.append(msg.get("content", ""))
                        # Only get the most recent director message
                        break

                # Add the director's instructions to the system prompt
                if director_instructions:
                    system_content += "\n\nRecent director instructions:\n" + "\n".join(director_instructions)

            task = current_input.get('task', 'No task specified')
            human_content = (
                f"Task: {task}\n\n"
                f"Context:\n{context}\n"
                f"{research_context}"
                f"\nManuscript section to review:\n{manuscript_excerpt[:5000] if manuscript_excerpt else 'No specific manuscript section provided'}"
            ).strip()
            if not human_content:
                human_content = "Please help with this writing project."

            # Combine system and human content into a single message
            combined_content = f"{system_content}\n\n{human_content}"

            # Create a single HumanMessage with the combined content
            user_message = HumanMessage(content=combined_content)

            # Setup LangSmith trace
            config = RunnableConfig()
            if langchain_tracer:
                config["callbacks"] = [langchain_tracer]
                config["tags"] = [f"agent:{agent_name}", f"project:{project_id}", f"phase:{state.get('phase', 'unknown')}"]
                # Set up metadata for the trace
                config["metadata"] = {
                    "agent_name": agent_name,
                    "project_id": project_id,
                    "phase": state.get("phase", "unknown"),
                    "task": task
                }

            # Create a streaming queue for real-time token updates
            stream_queue = queue.Queue(maxsize=1000)

            # Custom streaming callback to capture tokens in real-time with improved context
            class StreamingTokenCallback(StreamingStdOutCallbackHandler):
                def __init__(self):
                    super().__init__()
                    self.current_agent = agent_name
                    self.token_buffer = ""
                    self.last_token_time = time.time()

                def on_llm_start(self, serialized, prompts, **kwargs):
                    self.token_buffer = ""
                    self.last_token_time = time.time()
                    print(f"[Agent: {self.current_agent}] Starting processing...")

                def on_llm_new_token(self, token: str, **kwargs):
                    try:
                        # Add token to buffer and queue
                        self.token_buffer += token
                        if not stream_queue.full():
                            stream_queue.put(token)

                        # If we have a period/question mark/exclamation, or it's been more than 1 second,
                        # or buffer is getting large, log what we have
                        current_time = time.time()
                        if ('.' in token or '?' in token or '!' in token or
                                current_time - self.last_token_time > 1.0 or
                                len(self.token_buffer) > 80):

                            # Print with agent context
                            print(f"[Agent: {self.current_agent}] {self.token_buffer.strip()}")

                            # Reset buffer and time
                            self.token_buffer = ""
                            self.last_token_time = current_time

                    except Exception as e:
                        print(f"Error in streaming callback: {e}")

                def on_llm_end(self, response, **kwargs):
                    # Print any remaining tokens in buffer
                    if self.token_buffer:
                        print(f"[Agent: {self.current_agent}] {self.token_buffer.strip()}")
                        self.token_buffer = ""

                    print(f"[Agent: {self.current_agent}] Processing completed")

            # Add streaming callback to config
            streaming_handler = StreamingTokenCallback()
            if "callbacks" in config:
                config["callbacks"].append(streaming_handler)
            else:
                config["callbacks"] = [streaming_handler]

            # Define a helper function for model invocation with retry logic
            @retry_with_exponential_backoff(max_retries=2, initial_delay=1)
            def invoke_model_with_retry(model, messages, config):
                """Invoke model with retry and explicit CPU fallback"""
                try:
                    return model.invoke(messages, config=config)
                except Exception as e:
                    if "device" in str(e).lower() or "cuda" in str(e).lower() or "tensor" in str(e).lower():
                        print(f"Device-related error: {e}. Trying with CPU fallback.")
                        # Force CPU operation for this invocation
                        if hasattr(torch, "device"):
                            with torch.device("cpu"):
                                return model.invoke(messages, config=config)
                    raise  # Re-raise if not a device error or if CPU fallback failed

            # Add error handling around model invocation
            try:
                # Get the model (lazy loading) - use agent-specific model if configured
                model = self.get_model(agent_name)

                # Debug message for monitoring
                print(f"Invoking model for agent: {agent_name}")

                # Clear CUDA cache before invocation to reduce memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Try with simplified inputs if we encounter device issues
                response = None
                try:
                    # First try with retry mechanism
                    try:
                        response = invoke_model_with_retry(model, [user_message], config)
                        print(f"Model invocation completed for agent: {agent_name}")
                    except Exception as retry_error:
                        if "device" in str(retry_error).lower() or "tensor" in str(retry_error).lower():
                            # If we're having device-related issues, try a more direct approach with CPU
                            print(f"Trying direct CPU approach after device error: {str(retry_error)}")
                            with torch.device("cpu"):
                                # Create a simpler message to avoid complex tensor operations
                                simplified_message = HumanMessage(content=f"You are {agent_name}. Respond to: {task}")
                                response = model.invoke([simplified_message], config=config)
                        else:
                            raise retry_error

                except Exception as final_error:
                    # If even that fails, create a failure message
                    error_content = f"Error invoking model for {agent_name}: {str(final_error)}"
                    print(error_content)
                    response = AIMessage(content=error_content)

            except Exception as e:
                error_msg = f"Error invoking model for {agent_name}: {str(e)}"
                print(error_msg)
                traceback.print_exc()

                # Create a meaningful error response
                response = AIMessage(content=f"Error: {error_msg}\n\nI encountered a technical issue while processing your request. Please try again with a simpler query, or consider switching to a different model.")

            # Collect and format streamed tokens
            streamed_content = ""
            while not stream_queue.empty():
                try:
                    token = stream_queue.get(block=False)
                    streamed_content += token
                except queue.Empty:
                    break

            # Get the response content
            response_content = response.content if hasattr(response, 'content') else str(response)

            # Update state
            updated_state = state.copy()
            updated_state["messages"] = state.get("messages", []) + [
                {"role": "user", "content": combined_content},
                {"role": "assistant", "content": response_content}
            ]
            updated_state["count"] = state.get("count", 0) + 1
            updated_state["lnode"] = agent_name

            # Store streaming content for display in UI
            if "stream_content" not in updated_state:
                updated_state["stream_content"] = []
            if streamed_content:
                updated_state["stream_content"].append({
                    "agent": agent_name,
                    "content": streamed_content
                })

            # Apply any rewrites from content creation agents to update the manuscript
            if is_content_creation_agent(agent_name):
                updated_state = update_manuscript_with_rewrite(updated_state, agent_name, response_content)

            # Clean up memory after agent processing
            cleanup_memory()

            return updated_state

        return agent_function

    def update_model_config(self, new_config: Dict[str, Any]):
        """Update the model configuration for the AgentFactory."""
        if "default_model_config" in new_config:
            self.default_model_config.update(new_config["default_model_config"])
        if "agent_model_configs" in new_config:
            self.agent_model_configs.update(new_config["agent_model_configs"])

class StreamingTokenCallback(StreamingStdOutCallbackHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.token_buffer = ""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        """Handle the start of LLM generation."""
        self.token_buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process new tokens as they're generated."""
        if token:
            self.token_buffer += token
            if len(self.token_buffer) > 80 or any(p in token for p in '.!?'):
                try:
                    self.queue.put_nowait(self.token_buffer)
                    self.token_buffer = ""
                except queue.Full:
                    pass

    def on_llm_end(self, response, **kwargs) -> None:
        """Handle remaining tokens at generation end."""
        if self.token_buffer:
            try:
                self.queue.put_nowait(self.token_buffer)
            except queue.Full:
                pass
            self.token_buffer = ""

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

class SafeFallbackModel(BaseChatModel):
    """A safe fallback model that returns predefined responses when other models fail."""

    def __init__(self, agent_name_=None):
        super().__init__()
        self._agent_name = agent_name_ or "unknown"

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "safe_fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a safe fallback response."""
        try:
            # Create a contextual fallback message
            response = (
                f"I apologize, but I'm currently operating in fallback mode for {self._agent_name}. "
                "The main language model encountered an issue. Please try:\n"
                "1. Using a different model\n"
                "2. Reducing the input size\n"
                "3. Checking your system resources\n"
                "4. Restarting the application"
            )

            # Create message in expected format
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)

            # Return in the expected ChatResult format
            return ChatResult(generations=[generation])

        except Exception as e:
            # Ultimate fallback if even the safe response fails
            message = AIMessage(content=f"Critical error in fallback model: {str(e)}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async version of generate - returns same response."""
        return self._generate(messages, stop, run_manager, **kwargs)

class Storybook:
    def __init__(self, title: str, author: str, synopsis: str, manuscript: str):
        self.title = title
        self.author = author
        self.synopsis = synopsis
        self.manuscript = manuscript
        self.manuscript_chunks = self.split_manuscript(manuscript)

    def split_manuscript(self, manuscript: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[Dict[str, Any]]:
        """Split a manuscript into manageable chunks."""
        if not manuscript:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        texts = text_splitter.split_text(manuscript)
        chunks = []
        for i, text in enumerate(texts):
            chunks.append({
                "chunk_id": i,
                "content": text,
                "start_char": manuscript.find(text),
                "end_char": manuscript.find(text) + len(text),
            })

        return chunks

    def reassemble_manuscript(self) -> str:
        """Reassemble the full manuscript from chunks."""
        sorted_chunks = sorted(self.manuscript_chunks, key=lambda x: x.get("chunk_id", 0))
        full_manuscript = ""
        for chunk in sorted_chunks:
            content = chunk.get("content", "")
            if full_manuscript and not (full_manuscript.endswith("\n") or content.startswith("\n")):
                full_manuscript += "\n\n"
            full_manuscript += content
        return full_manuscript

    def update_chunk(self, chunk_id: int, new_content: str):
        """Update a specific chunk with new content."""
        for chunk in self.manuscript_chunks:
            if chunk["chunk_id"] == chunk_id:
                chunk["content"] = new_content
                break
        self.manuscript = self.reassemble_manuscript()

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        for chunk in self.manuscript_chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all manuscript chunks."""
        return self.manuscript_chunks

    def get_summary(self) -> str:
        """Get a summary of the storybook."""
        return f"Title: {self.title}\nAuthor: {self.author}\nSynopsis: {self.synopsis}\nTotal Chunks: {len(self.manuscript_chunks)}"

    def update_model_config(self, new_config: Dict[str, Any]):
        """Update the model configuration for the AgentFactory."""
        if "default_model_config" in new_config:
            self.default_model_config.update(new_config["default_model_config"])
        if "agent_model_configs" in new_config:
            self.agent_model_configs.update(new_config["agent_model_configs"])
