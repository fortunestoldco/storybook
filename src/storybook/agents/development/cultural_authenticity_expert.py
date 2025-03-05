from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.culture import (
    CulturalAuthenticityTool,
    RepresentationAnalysisTool,
    CulturalContextTool
)
from storybook.agents.base_agent import BaseAgent

class CulturalAuthenticityExpert(BaseAgent):
    """Expert responsible for cultural authenticity and representation."""
    
    def __init__(self):
        super().__init__(
            name="cultural_authenticity_expert",
            tools=[
                CulturalAuthenticityTool(),
                RepresentationAnalysisTool(),
                CulturalContextTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process cultural authenticity tasks."""
        task = state.current_input.get("task", {})
        
        if "context" in task.get("type", "").lower():
            context = await self.tools[2].invoke({
                "content": state.project.content,
                "culture": task.get("culture", "")
            })
            return {
                "messages": [AIMessage(content="Cultural context analysis completed")],
                "cultural_updates": {"context": context}
            }
            
        if "representation" in task.get("type", "").lower():
            analysis = await self.tools[1].invoke({
                "content": state.project.content,
                "context": task.get("context", {})
            })
            return {
                "messages": [AIMessage(content="Representation analysis completed")],
                "cultural_updates": {"representation": analysis}
            }
        
        authenticity = await self.tools[0].invoke({
            "content": state.project.content,
            "culture": task.get("culture", ""),
            "context": task.get("context", {})
        })
        return {
            "messages": [AIMessage(content="Cultural authenticity check completed")],
            "cultural_updates": {"authenticity": authenticity}
        }
