from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.quality import (
    QualityMetricsTool,
    QualityVerificationTool,
    QualityGateTool
)
from storybook.agents.base_agent import BaseAgent

class QualityAssessmentDirector(BaseAgent):
    """Director responsible for quality assessment and verification."""
    
    def __init__(self):
        super().__init__(
            name="quality_assessment_director",
            tools=[
                QualityMetricsTool(),
                QualityVerificationTool(),
                QualityGateTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        task = state.current_input.get("task", {})
        
        if "verification" in task.get("type", "").lower():
            verification = await self.tools[1].invoke({
                "content": state.project.content,
                "criteria": task.get("criteria", {})
            })
            return {
                "messages": [AIMessage(content="Quality verification completed")],
                "quality_updates": {"verification": verification}
            }
        
        if "gate" in task.get("type", "").lower():
            gate = await self.tools[2].invoke({
                "content": state.project.content,
                "gate_requirements": task.get("gate_requirements", {})
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