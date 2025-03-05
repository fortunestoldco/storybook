from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.character import (
    CharacterArcTool,
    ConsistencyCheckTool
)
from storybook.agents.base_agent import BaseAgent

class CharacterArcEvaluator(BaseAgent):
    """Specialist responsible for evaluating character arcs."""
    
    def __init__(self):
        super().__init__(
            name="character_arc_evaluator",
            tools=[
                CharacterArcTool(),
                ConsistencyCheckTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        task = state.current_input.get("task", {})
        
        if "consistency" in task.get("type", "").lower():
            check = await self.tools[1].invoke({
                "content": state.project.content,
                "character_id": task.get("character_id")
            })
            return {
                "messages": [AIMessage(content="Character consistency check completed")],
                "character_updates": {"consistency": check}
            }
        
        arc = await self.tools[0].invoke({
            "content": state.project.content,
            "character_id": task.get("character_id")
        })
        return {
            "messages": [AIMessage(content="Character arc analysis completed")],
            "character_updates": {"arc": arc}
        }
