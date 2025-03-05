from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.domain import (
    DomainResearchTool,
    FactVerificationTool,
    ExpertKnowledgeTool
)
from storybook.agents.base_agent import BaseAgent

class DomainKnowledgeSpecialist(BaseAgent):
    """Specialist responsible for domain-specific knowledge and research."""

    def __init__(self):
        super().__init__(
            name="domain_knowledge_specialist",
            tools=[
                DomainResearchTool(),
                FactVerificationTool(),
                ExpertKnowledgeTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process domain knowledge tasks."""
        task = state.current_input.get("task", {})
        domain = task.get("domain", "")

        if "verify" in task.get("type", "").lower():
            verification = await self.tools[1].arun(
                content=state.project.content,
                domain=domain,
                facts=task.get("facts", [])
            )
            return {
                "messages": [AIMessage(content="Facts verified")],
                "domain_updates": {"verification": verification}
            }

        if "expert" in task.get("type", "").lower():
            expertise = await self.tools[2].arun(
                content=state.project.content,
                domain=domain,
                query=task.get("query", "")
            )
            return {
                "messages": [AIMessage(content="Expert knowledge applied")],
                "domain_updates": {"expertise": expertise}
            }

        research = await self.tools[0].arun(
            content=state.project.content,
            domain=domain,
            research_params=task.get("parameters", {})
        )
        return {
            "messages": [AIMessage(content="Domain research completed")],
            "domain_updates": {"research": research}
        }
