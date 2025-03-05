from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from storybook.agents.base import BaseAgent
from .states import ResearchState, ResearchQuery, ResearchReport

class ResearchAgent(BaseAgent):
    """Base class for research-focused agents."""
    
    def __init__(self, name: str, tools: List[BaseTool], config: Dict[str, Any]):
        super().__init__(name, tools)
        self.research_config = config
        self.max_iterations = config.get("max_iterations", 3)
        self.quality_threshold = config.get("quality_threshold", 0.8)

    async def execute_research_cycle(self, state: ResearchState) -> Dict[str, Any]:
        """Execute a full research cycle."""
        while state["iterations"] < self.max_iterations:
            # Execute current queries
            results = await execute_research(state["queries"], self.research_config)
            
            # Compile report
            report = self.compile_report(results)
            
            # Analyze quality
            quality = analyze_research_quality(report)
            
            if quality["score"] >= self.quality_threshold:
                return {
                    "report": report,
                    "quality_score": quality["score"]
                }
                
            # Identify gaps and generate new queries
            gaps = identify_knowledge_gaps(report)
            new_queries = generate_followup_queries(gaps, state["context"])
            
            state["queries"] = new_queries
            state["iterations"] += 1
            
        return {
            "report": report,
            "quality_score": quality["score"]
        }