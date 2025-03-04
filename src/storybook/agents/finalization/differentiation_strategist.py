from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.marketing import (
    UniqueSellingPointTool,
    MarketDifferentiationTool,
    ValuePropositionTool
)
from storybook.agents.base_agent import BaseAgent

class DifferentiationStrategist(BaseAgent):
    """Strategist for identifying and enhancing unique selling points."""
    
    def __init__(self):
        super().__init__(
            name="differentiation_strategist",
            tools=[
                UniqueSellingPointTool(),
                MarketDifferentiationTool(),
                ValuePropositionTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Develop differentiation strategy."""
        task = state.current_input.get("task", "")
        
        if "market" in task.lower():
            differentiation = await self.tools[1].arun(
                content=state.project.content,
                market_analysis=state.project.content.get("market_analysis", {}),
                competitor_analysis=state.project.content.get("competitor_analysis", {})
            )
            return {
                "messages": [AIMessage(content="Market differentiation strategy developed")],
                "strategy_updates": {"differentiation": differentiation}
            }
        
        if "value" in task.lower():
            proposition = await self.tools[2].arun(
                content=state.project.content,
                unique_elements=state.project.content.get("unique_elements", []),
                target_audience=state.project.target_audience
            )
            return {
                "messages": [AIMessage(content="Value proposition defined")],
                "strategy_updates": {"value_proposition": proposition}
            }
        
        # Default to unique selling points
        usp = await self.tools[0].arun(
            content=state.project.content,
            genre=state.project.genre,
            market_position=state.project.content.get("market_position", {})
        )
        
        return {
            "messages": [AIMessage(content="Unique selling points identified")],
            "strategy_updates": {"usp": usp}
        }