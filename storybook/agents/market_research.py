from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from storybook.agents.base import BaseAgent
from storybook.state import State, AgentOutput

logger = logging.getLogger(__name__)

class MarketResearchAgent(BaseAgent):
    """Agent responsible for market research and analysis."""

    async def process_manuscript(self, state: State) -> Dict[str, Any]:
        """Process manuscript for market analysis."""
        try:
            manuscript_state = state.get_manuscript_state()
            
            # Market analysis prompt
            prompt = ChatPromptTemplate.from_template(
                """As a publishing industry analyst, research the current market for this manuscript.
                Analyze:
                1. Current bestseller trends
                2. Reader demographics and preferences 
                3. Market saturation and opportunities
                4. Pricing and format trends
                
                Manuscript:
                {manuscript}
                
                Title: {title}
                
                Provide detailed market analysis in JSON format.
                """
            )

            # Create and run the chain
            chain = prompt | self.llm | StrOutputParser()
            
            analysis = await chain.ainvoke({
                "manuscript": manuscript_state.manuscript,
                "title": manuscript_state.title
            })

            return AgentOutput(
                content=analysis,
                timestamp=datetime.now(),
                agent_id="market_research"
            ).dict()

        except Exception as e:
            logger.error(f"Market research failed: {str(e)}")
            return self.handle_error(e)
