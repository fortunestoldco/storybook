from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.grammar import (
    GrammarCheckTool,
    StyleConsistencyTool,
    UsageVerificationTool
)
from storybook.agents.base_agent import BaseAgent

class GrammarConsistencyChecker(BaseAgent):
    """Checker responsible for grammatical consistency."""
    
    def __init__(self):
        super().__init__(
            name="grammar_consistency_checker",
            tools=[
                GrammarCheckTool(),
                StyleConsistencyTool(),
                UsageVerificationTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Check and maintain grammatical consistency."""
        task = state.current_input.get("task", "")
        section_id = task.get("section_id", "")
        
        if "style" in task.lower():
            style_check = await self.tools[1].arun(
                content=task.get("content", ""),
                style_guide=state.project.style_preferences
            )
            return {
                "messages": [AIMessage(content="Style consistency checked")],
                "grammar_updates": {section_id: {"style": style_check}}
            }
        
        if "usage" in task.lower():
            usage = await self.tools[2].arun(
                content=task.get("content", ""),
                terms=task.get("terms", [])
            )
            return {
                "messages": [AIMessage(content="Usage verified")],
                "grammar_updates": {section_id: {"usage": usage}}
            }
        
        # Default to grammar check
        grammar = await self.tools[0].arun(
            content=task.get("content", ""),
            rules=state.project.style_preferences.get("grammar_rules", {})
        )
        
        return {
            "messages": [AIMessage(content="Grammar checked")],
            "grammar_updates": {section_id: {"grammar": grammar}}
        }
