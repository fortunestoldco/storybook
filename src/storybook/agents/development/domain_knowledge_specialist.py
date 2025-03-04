from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.domain import (
    ResearchTool,
    FactCheckingTool,
    TerminologyManagerTool
)
from storybook.agents.base_agent import BaseAgent

class DomainKnowledgeSpecialist(BaseAgent):
    """Specialist responsible for domain-specific knowledge and accuracy."""
    
    def __init__(self):
        super().__init__(
            name="domain_knowledge_specialist",
            tools=[
                ResearchTool(),
                FactCheckingTool(),
                TerminologyManagerTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process domain knowledge requirements."""
        task = state.current_input.get("task", "")
        domain = task.get("domain", "")
        
        if "fact_check" in task.lower():
            verification = await self.tools[1].arun(
                content=state.project.content,
                domain=domain,
                claims=task.get("claims", [])
            )
            return {
                "messages": [AIMessage(content=f"Facts verified for domain: {domain}")],
                "domain_updates": {"fact_check": verification}
            }
        
        if "terminology" in task.lower():
            terms = await self.tools[2].arun(
                domain=domain,
                context=state.project.content,
                term_list=task.get("terms", [])
            )
            return {
                "messages": [AIMessage(content=f"Terminology updated for: {domain}")],
                "domain_updates": {"terminology": terms}
            }
        
        # Default to research
        research = await self.tools[0].arun(
            domain=domain,
            query=task.get("query", ""),
            context=state.project.content
        )
        
        return {
            "messages": [AIMessage(content=f"Research completed for: {domain}")],
            "domain_updates": {"research": research}
        }