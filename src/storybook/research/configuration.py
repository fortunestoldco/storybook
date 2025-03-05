from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict

class Configuration:
    """Configuration for research operations."""
    
    def __init__(self, 
                search_api: str = "tavily", 
                search_api_config: Optional[Dict[str, Any]] = None,
                mongo_connection_string: Optional[str] = None,
                mongo_database_name: Optional[str] = None,
                queries_per_iteration: int = 3,
                max_iterations: int = 3,
                quality_threshold: float = 0.7,
                agent_model_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize research configuration.
        
        Args:
            search_api: API to use for searches
            search_api_config: Configuration for search API
            mongo_connection_string: MongoDB connection string
            mongo_database_name: MongoDB database name
            queries_per_iteration: Number of queries per iteration
            max_iterations: Maximum number of iterations
            quality_threshold: Threshold for research quality
            agent_model_configs: Model configurations for agents
        """
        self.search_api = search_api
        self.search_api_config = search_api_config or {}
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database_name = mongo_database_name
        self.queries_per_iteration = queries_per_iteration
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.agent_model_configs = agent_model_configs or {
            "research_quality_analyzer": {
                "provider": "anthropic", 
                "model_name": "claude-3-sonnet-20240229"
            },
            "research_gap_analyzer": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229"
            },
            "research_query_generator": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229"
            }
        }
    
    @staticmethod
    def from_runnable_config(config: Dict[str, Any]) -> 'Configuration':
        """Create configuration from runnable config.
        
        Args:
            config: Runnable configuration
            
        Returns:
            Research configuration
        """
        if not config:
            return Configuration()
        
        configurable = config.get("configurable", {})
        
        return Configuration(
            search_api=configurable.get("search_api", "tavily"),
            search_api_config=configurable.get("search_api_config"),
            mongo_connection_string=configurable.get("mongodb_connection_string"),
            mongo_database_name=configurable.get("mongodb_database_name"),
            queries_per_iteration=configurable.get("queries_per_iteration", 3),
            max_iterations=configurable.get("max_iterations", 3),
            quality_threshold=configurable.get("quality_threshold", 0.7),
            agent_model_configs=configurable.get("agent_model_configs")
        )