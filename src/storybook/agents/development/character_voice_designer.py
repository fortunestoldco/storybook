from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.character import (
    VoicePatternTool,
    DialogueStyleTool,
    ExpressionAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class CharacterVoiceDesigner(BaseAgent):
    """Designer responsible for character voice and dialogue patterns."""
    
    def __init__(self):
        super().__init__(
            name="character_voice_designer",
            tools=[
                VoicePatternTool(),
                DialogueStyleTool(),
                ExpressionAnalysisTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Design and maintain character voices."""
        task = state.current_input.get("task", "")
        character_id = task.get("character_id", "")
        
        if "dialogue" in task.lower():
            style = await self.tools[1].arun(
                character_id=character_id,
                profile=state.project.content.get("characters", {}).get(character_id, {}),
                context=task.get("context", {})
            )
            return {
                "messages": [AIMessage(content=f"Dialogue style updated: {character_id}")],
                "voice_updates": {
                    character_id: {"dialogue_style": style}
                }
            }
            
        if "expression" in task.lower():
            analysis = await self.tools[2].arun(
                character_id=character_id,
                dialogue=task.get("dialogue", ""),
                voice_pattern=state.project.content.get("characters", {})
                    .get(character_id, {}).get("voice_pattern", {})
            )
            return {
                "messages": [AIMessage(content=f"Expression analysis complete: {character_id}")],
                "voice_updates": {
                    character_id: {"expression_analysis": analysis}
                }
            }
        
        # Default to voice pattern generation
        pattern = await self.tools[0].arun(
            character_id=character_id,
            profile=state.project.content.get("characters", {}).get(character_id, {}),
            genre=state.project.genre
        )
        
        return {
            "messages": [AIMessage(content=f"Voice pattern updated: {character_id}")],
            "voice_updates": {
                character_id: {"voice_pattern": pattern}
            }
        }