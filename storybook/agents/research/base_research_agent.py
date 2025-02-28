from typing import Dict, Any, Optional
from ..base_agent import BaseAgent
from .utils import ResearchUtils

class BaseResearchAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.research_utils = ResearchUtils(config)
        
    async def research_topic(self, query: str, collection_name: str) -> Dict[str, Any]:
        """Conduct research on a specific topic."""
        # Search and store results
        results = await self.research_utils.search_and_store(query, collection_name)
        
        # Analyze coverage
        analysis = self.research_utils.analyze_coverage(results["results"], self.llm)
        
        return {
            "results": results["results"],
            "analysis": analysis,
            "collection": collection_name
        }