from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.editorial import (
    EditorialPlanningTool,
    QualityAssessmentTool,
    RevisionCoordinationTool
)
from storybook.agents.base_agent import BaseAgent

class EditorialDirector(BaseAgent):
    """Director responsible for managing the editorial process."""
    
    def __init__(self):
        super().__init__(
            name="editorial_director",
            tools=[
                EditorialPlanningTool(),
                QualityAssessmentTool(),
                RevisionCoordinationTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Manage editorial process and coordinate revisions."""
        task = state.current_input.get("task", "")
        
        if "quality" in task.lower():
            assessment = await self.tools[1].arun(
                content=state.project.content,
                quality_standards=state.project.quality_assessment,
                editorial_goals=state.project.content.get("editorial_goals", {})
            )
            return {
                "messages": [AIMessage(content="Quality assessment complete")],
                "editorial_updates": {"quality_assessment": assessment}
            }
        
        if "revision" in task.lower():
            coordination = await self.tools[2].arun(
                content=state.project.content,
                revision_notes=task.get("revision_notes", []),
                agents=state.project.content.get("active_agents", [])
            )
            return {
                "messages": [AIMessage(content="Revision tasks coordinated")],
                "editorial_updates": {"revision_plan": coordination}
            }
        
        # Default to editorial planning
        plan = await self.tools[0].arun(
            content=state.project.content,
            current_phase=state.phase,
            quality_metrics=state.project.quality_assessment
        )
        
        return {
            "messages": [AIMessage(content="Editorial plan updated")],
            "editorial_updates": {"plan": plan}
        }