from typing import Dict, Any, List, Tuple, Set, Optional
import spacy
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from pydantic import BaseModel, Field
from collections import defaultdict
import logging
import numpy as np
import os
from config.llm_config import LLMConfig, NLPConfig
import asyncio
from functools import lru_cache

class NLPProcessor:
    """Handles NLP operations for consistency checking using remote Ollama endpoint."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.llm_config = LLMConfig(
            base_url=os.getenv('OLLAMA_BASE_URL', 'https://your-ollama-endpoint.com')
        )
        self.nlp_config = NLPConfig()
        
        # Initialize caching if enabled
        if self.llm_config.enable_cache:
            set_llm_cache(InMemoryCache())
        
        # Initialize models
        self._init_models()
        
        # Load SpaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Error loading SpaCy model: {str(e)}")
            raise

        # Initialize prompt templates
        self._init_prompts()

    def _init_models(self):
        """Initialize different LLM models for specific tasks."""
        self.models = {}
        
        try:
            for task, model_type in self.llm_config.task_models.items():
                model_config = self.llm_config.models[model_type]
                
                self.models[task] = Ollama(
                    base_url=self.llm_config.base_url,
                    model=model_config["name"],
                    temperature=model_config.get("temperature", 0.1),
                    top_p=model_config.get("top_p", 0.9)
                )
            
            # Initialize embeddings model
            self.embeddings = OllamaEmbeddings(
                base_url=self.llm_config.base_url,
                model=self.llm_config.models["embeddings"]["name"]
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    async def _get_model_for_task(self, task: str) -> Ollama:
        """Get the appropriate model for a specific task with fallback."""
        try:
            return self.models[task]
        except KeyError:
            self.logger.warning(f"No specific model for task {task}, using fallback")
            return self.models.get("fallback")

    @lru_cache(maxsize=1000)
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using cached embeddings."""
        try:
            emb1 = await self.embeddings.aembed_query(text1)
            emb2 = await self.embeddings.aembed_query(text2)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0

    async def analyze_with_retries(
        self,
        task: str,
        prompt: str,
        parser: PydanticOutputParser,
        max_retries: Optional[int] = None
    ) -> Any:
        """Execute analysis with automatic retries."""
        retries = max_retries or self.llm_config.max_retries
        
        for attempt in range(retries):
            try:
                model = await self._get_model_for_task(task)
                response = await model.agenerate([prompt])
                return parser.parse(response.generations[0].text)
                
            except Exception as e:
                if attempt == retries - 1:
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(self.llm_config.retry_delay * (attempt + 1))

    async def analyze_character_traits(
        self,
        text: str,
        character_data: Dict[str, Any]
    ) -> TraitAnalysis:
        """Analyze character traits using the linguistic model."""
        try:
            prompt = self.trait_prompt.format(
                text=text,
                character_name=character_data.get("name", ""),
                previous_traits=str(character_data.get("traits", {}))
            )
            
            return await self.analyze_with_retries(
                "character_analysis",
                prompt,
                PydanticOutputParser(pydantic_object=TraitAnalysis)
            )
            
        except Exception as e:
            self.logger.error(f"Error in character trait analysis: {str(e)}")
            return TraitAnalysis(traits=[], contradictions=[], confidence=0.0)

    async def analyze_location(
        self,
        text: str,
        location_data: Dict[str, Any]
    ) -> LocationAnalysis:
        """Analyze location descriptions and check for inconsistencies."""
        try:
            prompt = self.location_prompt.format(
                text=text,
                location_name=location_data.get("name", ""),
                previous_description=location_data.get("description", "")
            )
            
            response = await self.llm.agenerate([prompt])
            parser = PydanticOutputParser(pydantic_object=LocationAnalysis)
            return parser.parse(response.generations[0].text)
            
        except Exception as e:
            self.logger.error(f"Error in location analysis: {str(e)}")
            return LocationAnalysis(properties=[], inconsistencies=[], confidence=0.0)

    async def analyze_plot_point(
        self,
        text: str,
        required_elements: Dict[str, Any]
    ) -> PlotAnalysis:
        """Analyze plot points and verify required elements."""
        try:
            prompt = self.plot_prompt.format(
                text=text,
                required_elements=str(required_elements)
            )
            
            response = await self.llm.agenerate([prompt])
            parser = PydanticOutputParser(pydantic_object=PlotAnalysis)
            return parser.parse(response.generations[0].text)
            
        except Exception as e:
            self.logger.error(f"Error in plot point analysis: {str(e)}")
            return PlotAnalysis(found_elements=[], missing_elements=[], confidence=0.0)

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using Ollama embeddings."""
        try:
            # Get embeddings
            emb1 = await self.embeddings.aembed_query(text1)
            emb2 = await self.embeddings.aembed_query(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0