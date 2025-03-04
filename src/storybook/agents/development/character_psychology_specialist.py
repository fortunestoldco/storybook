from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.character import (
    PsychologyProfileTool,
    MotivationAnalysisTool,
    ConflictResponseTool
)
from storybook.agents.base_agent import BaseAgent

class CharacterPsychologySpecialist(BaseAgent):
    """Specialist in developing psychological profiles for characters."""
    
    def __init__(self):
        super().__init__(
            name="character_psychology_specialist",
            tools=[
                PsychologyProfileTool(),
                MotivationAnalysisTool(),
                ConflictResponseTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Develop and maintain character psychology."""
        task = state.current_input.get("task", "")
        character_id = task.get("character_id", "")
        
        if "motivation" in task.lower():
            motivation = await self.tools[1].arun(
                character_id=character_id,
                content=state.project.content
            )
            return {
                "messages": [AIMessage(content=f"Character motivation analyzed: {character_id}")],
                "character_updates": {
                    character_id: {"motivation": motivation}
                }
            }
        
        if "conflict" in task.lower():
            response = await self.tools[2].arun(
                character_id=character_id,
                conflict=task.get("conflict", {}),
                profile=state.project.content.get("characters", {}).get(character_id, {})
            )
            return {
                "messages": [AIMessage(content=f"Conflict response analyzed: {character_id}")],
                "character_updates": {
                    character_id: {"conflict_response": response}
                }
            }
        
        # Default to psychology profile
        profile = await self.tools[0].arun(
            character_id=character_id,
            content=state.project.content
        )
        
        return {
            "messages": [AIMessage(content=f"Character profile updated: {character_id}")],
            "character_updates": {
                character_id: {"profile": profile}
            }
        }