from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.quality import (
    QualityMetricsTool,
    QualityGateTool,
    QualityVerificationTool
)
from storybook.agents.base_agent import BaseAgent

class QualityAssessmentDirector(BaseAgent):
    """Director responsible for quality assessment and control."""
    
    def __init__(self):
        super().__init__(
            name="quality_assessment_director",
            tools=[
                QualityMetricsTool(),
                QualityGateTool(),
                QualityVerificationTool()
            ]
        )
        self._validate_tools()

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process quality assessment tasks."""
        task = state.current_input.get("task", {})
        
        if "verification" in task.get("type", "").lower():
            verification = await self.tools[2].invoke({
                "content": state.project.content,
                "criteria": task.get("criteria", {})
            })
            return {
                "messages": [AIMessage(content="Quality verification completed")],
                "quality_updates": {"verification": verification}
            }
            
        if "gate" in task.get("type", "").lower():
            gate = await self.tools[1].invoke({
                "content": state.project.content,
                "threshold": task.get("threshold", 0.8)
            })
            return {
                "messages": [AIMessage(content="Quality gate check completed")],
                "quality_updates": {"gate": gate}
            }
        
        metrics = await self.tools[0].invoke({
            "content": state.project.content,
            "metric_types": task.get("metric_types", [])
        })
        return {
            "messages": [AIMessage(content="Quality metrics assessment completed")],
            "quality_updates": {"metrics": metrics}
        }
