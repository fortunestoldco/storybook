from typing import Dict, Any, Optional
from ..base_agent import BaseAgent

class CreativeCoordinator(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.system_prompt = """You are a Creative Coordinator responsible for 
        translating architectural decisions into creative development tasks."""

    async def process_creative_phase(
        self, 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        architecture = state.get("structure", {})
        research = state.get("research_findings", {})
        
        development_plan = self.generate_content(
            self.system_prompt,
            f"""Plan creative development based on architectural decisions 
            and research findings."""
        )
        
        return {
            "development_plan": development_plan,
            "creative_tasks": self._generate_tasks(development_plan),
            "research_requirements": self._identify_research_needs(development_plan)
        }