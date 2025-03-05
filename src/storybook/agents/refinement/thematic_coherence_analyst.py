from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.theme import (
    ThemeIdentificationTool,
    MotifAnalysisTool,
    SymbolismTool
)
from storybook.agents.base_agent import BaseAgent

class ThematicCoherenceAnalyst(BaseAgent):
    """Analyst responsible for maintaining thematic consistency."""
    
    def __init__(self):
        super().__init__(
            name="thematic_coherence_analyst",
            tools=[
                ThemeIdentificationTool(),
                MotifAnalysisTool(),
                SymbolismTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Analyze and maintain thematic elements."""
        task = state.current_input.get("task", "")
        
        if "motif" in task.lower():
            motifs = await self.tools[1].arun(
                content=state.project.content,
                themes=state.project.content.get("themes", []),
                section_id=task.get("section_id", "")
            )
            return {
                "messages": [AIMessage(content="Motif analysis completed")],
                "theme_updates": {"motifs": motifs}
            }
        
        if "symbolism" in task.lower():
            symbols = await self.tools[2].arun(
                content=state.project.content,
                symbols=state.project.content.get("symbols", {}),
                themes=state.project.content.get("themes", [])
            )
            return {
                "messages": [AIMessage(content="Symbolism analysis completed")],
                "theme_updates": {"symbols": symbols}
            }
        
        # Default to theme identification
        themes = await self.tools[0].arun(
            content=state.project.content,
            existing_themes=state.project.content.get("themes", []),
            genre=state.project.genre
        )
        
        return {
            "messages": [AIMessage(content="Themes identified and analyzed")],
            "theme_updates": {"themes": themes}
        }
