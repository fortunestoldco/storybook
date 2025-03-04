from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.quality import QualityMetricsTool, QualityGateTool
from storybook.agents.base_agent import BaseAgent

class QualityAssessmentDirector(BaseAgent):
    """Director for assessing and maintaining quality standards."""
    
    def __init__(self):
        super().__init__(
            name="quality_assessment_director",
            tools=[
                QualityMetricsTool(),
                QualityGateTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Assess quality metrics and manage quality gates."""
        task = state.current_input.get("task", "")
        
        if "gate" in task.lower():
            gate_result = await self.tools[1].arun(
                phase=state.phase,
                metrics=state.project.quality_assessment
            )
            return {
                "messages": [
                    AIMessage(content=f"Quality gate assessment: {gate_result}")
                ],
                "gate_result": gate_result
            }
        
        metrics = await self.tools[0].arun(
            content=state.project.content,
            phase=state.phase
        )
        
        return {
            "messages": [
                AIMessage(content=f"Quality metrics updated: {metrics}")
            ],
            "quality_metrics": metrics
        }