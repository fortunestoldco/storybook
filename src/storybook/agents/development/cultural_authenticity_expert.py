from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.culture import (
    CulturalAnalysisTool,
    SensitivityCheckTool,
    RepresentationTool
)
from storybook.agents.base_agent import BaseAgent

class CulturalAuthenticityExpert(BaseAgent):
    """Expert ensuring cultural authenticity and sensitivity."""
    
    def __init__(self):
        super().__init__(
            name="cultural_authenticity_expert",
            tools=[
                CulturalAnalysisTool(),
                SensitivityCheckTool(),
                RepresentationTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Ensure cultural authenticity and appropriate representation."""
        task = state.current_input.get("task", "")
        culture = task.get("culture", "")
        
        if "sensitivity" in task.lower():
            check = await self.tools[1].arun(
                content=state.project.content,
                culture=culture,
                context=task.get("context", {})
            )
            return {
                "messages": [AIMessage(content=f"Sensitivity check completed for: {culture}")],
                "cultural_updates": {"sensitivity": check}
            }
        
        if "representation" in task.lower():
            analysis = await self.tools[2].arun(
                content=state.project.content,
                culture=culture,
                characters=state.project.content.get("characters", {})
            )
            return {
                "messages": [AIMessage(content=f"Representation analysis for: {culture}")],
                "cultural_updates": {"representation": analysis}
            }
        
        # Default to cultural analysis
        cultural_elements = await self.tools[0].arun(
            content=state.project.content,
            culture=culture,
            elements=task.get("elements", [])
        )
        
        return {
            "messages": [AIMessage(content=f"Cultural analysis completed for: {culture}")],
            "cultural_updates": {"analysis": cultural_elements}
        }