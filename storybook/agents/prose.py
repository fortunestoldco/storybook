from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from storybook.agents.base import BaseAgent
from storybook.state import State, AgentOutput

logger = logging.getLogger(__name__)

class ProseSpecialist(BaseAgent):
    """Agent that enhances prose quality and style."""

    async def process_manuscript(self, state: State) -> Dict[str, Any]:
        """Process manuscript for prose enhancement."""
        try:
            manuscript_state = state.get_manuscript_state()
            
            # Prose analysis prompt
            prompt = ChatPromptTemplate.from_template(
                """Analyze and elevate the prose quality of this manuscript.
                Focus on:
                1. Sentence structure and variation
                2. Word choice precision and impact
                3. Imagery and sensory details
                4. Show vs. tell balance
                5. Voice consistency and distinctiveness
                6. Paragraph flow and transitions
                
                Manuscript:
                {manuscript}
                
                Title: {title}
                
                Style Guidelines:
                {style_guide}
                
                Provide detailed prose analysis in JSON format including:
                - Style assessment
                - Language patterns
                - Enhancement opportunities
                - Specific recommendations
                - Before/after examples
                """
            )

            # Create and run the chain
            chain = prompt | self.llm | StrOutputParser()
            
            analysis = await chain.ainvoke({
                "manuscript": manuscript_state.manuscript,
                "title": manuscript_state.title,
                "style_guide": state.language.content if state.language else {}
            })

            return AgentOutput(
                content=analysis,
                timestamp=datetime.now(),
                agent_id="prose_elevation_specialist"
            ).dict()

        except Exception as e:
            logger.error(f"Prose analysis failed: {str(e)}")
            return self.handle_error(e)
