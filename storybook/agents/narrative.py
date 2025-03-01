from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from storybook.agents.base import BaseAgent
from storybook.state import State, AgentOutput

logger = logging.getLogger(__name__)

class NarrativeArcSurgeon(BaseAgent):
    """Agent that analyzes and enhances narrative structure."""

    async def process_manuscript(self, state: State) -> Dict[str, Any]:
        """Process manuscript for narrative structure analysis."""
        try:
            manuscript_state = state.get_manuscript_state()
            
            # Narrative analysis prompt
            prompt = ChatPromptTemplate.from_template(
                """Analyze the narrative structure of this manuscript with surgical precision.
                Identify and evaluate:
                1. Plot structure and major turning points
                2. Pacing and tension graph
                3. Subplot integration patterns
                4. Scene sequencing effectiveness
                5. Hook and resolution strategy
                
                Manuscript:
                {manuscript}
                
                Title: {title}
                
                Provide detailed analysis with specific recommendations in JSON format.
                Include:
                - Structural assessment
                - Pacing analysis
                - Scene breakdown
                - Recommendations for improvement
                - Tension mapping
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
                agent_id="narrative"
            ).dict()

        except Exception as e:
            logger.error(f"Narrative analysis failed: {str(e)}")
            return self.handle_error(e)
