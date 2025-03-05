from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.dialogue.refinement import (
    ConversationFlowTool,
    DialoguePolishingTool,
    SubtextEnhancementTool
)
from storybook.agents.base_agent import BaseAgent

class DialogueRefinementExpert(BaseAgent):
    """Expert responsible for refining and polishing dialogue."""
    
    def __init__(self):
        super().__init__(
            name="dialogue_refinement_expert",
            tools=[
                ConversationFlowTool(),
                DialoguePolishingTool(),
                SubtextEnhancementTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process dialogue refinement tasks."""
        task = state.current_input.get("task", {})
        
        if "subtext" in task.get("type", "").lower():
            subtext = await self.tools[2].invoke({
                "content": state.project.content,
                "dialogue_id": task.get("dialogue_id")
            })
            return {
                "messages": [AIMessage(content="Subtext enhancement completed")],
                "dialogue_updates": {"subtext": subtext}
            }
            
        if "polish" in task.get("type", "").lower():
            polish = await self.tools[1].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Dialogue polishing completed")],
                "dialogue_updates": {"polish": polish}
            }
        
        flow = await self.tools[0].invoke({
            "content": state.project.content,
            "scene_id": task.get("scene_id")
        })
        return {
            "messages": [AIMessage(content="Conversation flow analysis completed")],
            "dialogue_updates": {"flow": flow}
        }