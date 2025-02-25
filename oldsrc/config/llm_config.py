from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import asyncio
import aiohttp
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

class LLMConfig(BaseModel):
    """Configuration for different LLM models used in NLP tasks."""
    
    base_url: str = "https://your-ollama-endpoint.com"
    
    models: Dict[str, Dict[str, Any]] = {
        "analysis": {
            "name": "mixtral",
            "temperature": 0.1,
            "top_p": 0.9,
            "context_window": 32768,
            "description": "Primary model for deep analysis tasks"
        },
        "embeddings": {
            "name": "nomic-embed-text",
            "temperature": 0.0,
            "description": "Specialized model for text embeddings"
        },
        "code_analysis": {
            "name": "codellama:34b",
            "temperature": 0.1,
            "top_p": 0.9,
            "description": "Specialized for code-related analysis"
        },
        "linguistic": {
            "name": "yi:34b",
            "temperature": 0.2,
            "top_p": 0.9,
            "description": "Specialized for linguistic analysis"
        },
        "fallback": {
            "name": "llama2:70b",
            "temperature": 0.1,
            "description": "Fallback model for general tasks"
        }
    }
    
    task_models: Dict[str, str] = {
        "character_analysis": "linguistic",
        "plot_analysis": "analysis",
        "location_analysis": "analysis",
        "consistency_check": "analysis",
        "code_review": "code_analysis",
        "semantic_similarity": "embeddings",
        "theme_analysis": "analysis",
        "structure_analysis": "analysis"
    }
    
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_ttl: int = 3600

class LLMRouter:
    def __init__(self, tools_service, config: Optional[LLMConfig] = None):
        self.tools_service = tools_service
        self.config = config or LLMConfig()
        self.gpu_enabled = os.getenv("GPU", "false").lower() == "true"
        self.cloud_api_key = os.getenv("CLOUD_API_KEY")
        self.cloud_api_url = os.getenv("CLOUD_API_URL", self.config.base_url)
        
        if self.config.enable_cache:
            set_llm_cache(InMemoryCache())

    async def get_llm(self, task: str) -> Any:
        model_type = self.config.task_models.get(task, "fallback")
        model_config = self.config.models[model_type]
        
        if self.gpu_enabled:
            return Ollama(
                base_url=self.tools_service.config.get("ollama_base_url", self.config.base_url),
                model=model_config["name"],
                temperature=model_config.get("temperature", 0.1),
                top_p=model_config.get("top_p", 0.9)
            )
        else:
            return ChatOpenAI(
                api_key=self.cloud_api_key,
                base_url=self.cloud_api_url,
                model_kwargs=model_config
            )

    async def process_with_streaming(self, 
                                   task: str,
                                   prompt: str, 
                                   parser: Optional[PydanticOutputParser] = None) -> Any:
        model_type = self.config.task_models.get(task, "fallback")
        model_config = self.config.models[model_type]
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.max_retries):
                try:
                    if self.gpu_enabled:
                        llm = await self.get_llm(task)
                        response = await llm.agenerate([prompt])
                        text = response.generations[0].text
                    else:
                        headers = {
                            "Authorization": f"Bearer {self.cloud_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        data = {
                            "prompt": prompt,
                            "stream": True,
                            **model_config
                        }
                        
                        async with session.post(
                            self.cloud_api_url,
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status >= 400:
                                raise aiohttp.ClientError(
                                    f"API error: {response.status} - {await response.text()}"
                                )
                            
                            text = ""
                            async for chunk in response.content.iter_chunked(1024):
                                text += chunk.decode()
                                
                            if not text:
                                raise ValueError("Empty response from API")
                    
                    return parser.parse(text) if parser else text
                    
                except (aiohttp.ClientError, ValueError) as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

class NLPConfig(BaseModel):
    similarity_thresholds: Dict[str, float] = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
    
    confidence_thresholds: Dict[str, float] = {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.5
    }
    
    analysis_parameters: Dict[str, Any] = {
        "max_context_length": 4096,
        "overlap_window": 256,
        "batch_size": 10
    }