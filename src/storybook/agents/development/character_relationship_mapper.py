from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.character import (
    RelationshipGraphTool,
    DynamicsAnalysisTool,
    ConflictMapTool
)
from storybook.agents.base_agent import BaseAgent

class CharacterRelationshipMapper(BaseAgent):
    """Mapper responsible for character relationships and dynamics."""
    
    def __init__(self):
        super().__init__(
            name="character_relationship_mapper",
            tools=[
                RelationshipGraphTool(),
                DynamicsAnalysisTool(),
                ConflictMapTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Map and analyze character relationships."""
        task = state.current_input.get("task", "")
        characters = state.project.content.get("characters", {})
        
        if "dynamics" in task.lower():
            analysis = await self.tools[1].arun(
                characters=characters,
                relationship_graph=state.project.content.get("relationship_graph", {}),
                context=task.get("context", {})
            )
            return {
                "messages": [AIMessage(content="Character dynamics analyzed")],
                "relationship_updates": {"dynamics": analysis}
            }
            
        if "conflict" in task.lower():
            conflict_map = await self.tools[2].arun(
                characters=characters,
                relationship_graph=state.project.content.get("relationship_graph", {}),
                plot_threads=state.project.content.get("plot_threads", [])
            )
            return {
                "messages": [AIMessage(content="Relationship conflicts mapped")],
                "relationship_updates": {"conflicts": conflict_map}
            }
        
        # Default to relationship graph update
        graph = await self.tools[0].arun(
            characters=characters,
            plot_threads=state.project.content.get("plot_threads", [])
        )
        
        return {
            "messages": [AIMessage(content="Relationship graph updated")],
            "relationship_updates": {"graph": graph}
        }