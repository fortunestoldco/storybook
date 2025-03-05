from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.market.analysis import (
    MarketAnalysisTool,
    PositioningStrategyTool,
    CompetitorAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class MarketAlignmentDirector(BaseAgent):
    """Director responsible for market alignment."""
    
    def __init__(self):
        super().__init__(
            name="market_alignment_director",
            tools=[
                MarketAnalysisTool(),
                PositioningStrategyTool(),
                CompetitorAnalysisTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process market alignment tasks."""
        task = state.current_input.get("task", "")
        project = state.project
        
        if "position" in task.lower():
            position = await self.tools[1].arun(
                content=project.content,
                market_analysis=project.content.get("market_analysis", {})
            )
            return {
                "messages": [AIMessage(content="Market positioning strategy developed")],
                "market_updates": {"positioning_strategy": position}
            }
            
        if "competitor" in task.lower():
            competitors = await self.tools[2].arun(
                genre=project.genre[0] if project.genre else "",
                target_market={"audience": project.target_audience}
            )
            return {
                "messages": [AIMessage(content="Competitor analysis completed")],
                "market_updates": {"competitor_analysis": competitors}
            }
        
        # Default to market analysis
        analysis = await self.tools[0].arun(
            genre=project.genre[0] if project.genre else "",
            target_audience={"audience": project.target_audience}
        )
        return {
            "messages": [AIMessage(content="Market analysis completed")],
            "market_updates": {"market_analysis": analysis}
        }
