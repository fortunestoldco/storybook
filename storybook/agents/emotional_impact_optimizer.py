from typing import Dict, Any, Optional
import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from storybook.agents.base import BaseAgent
from storybook.state import State, AgentOutput

logger = logging.getLogger(__name__)

class EmotionalImpactOptimizer(BaseAgent):
    """Agent that analyzes and enhances emotional resonance."""

    async def process_manuscript(self, state: State) -> Dict[str, Any]:
        """Process manuscript for emotional impact analysis."""
        try:
            manuscript_state = state.get_manuscript_state()
            
            # Emotional impact analysis prompt
            prompt = ChatPromptTemplate.from_template(
                """Analyze the emotional impact and resonance of this manuscript.
                Evaluate:
                1. Emotional arcs for each major character
                2. Scene-level emotional beats
                3. Reader emotional journey mapping
                4. Catharsis points and buildup
                5. Emotional authenticity
                
                Manuscript:
                {manuscript}
                
                Title: {title}
                
                Character Analysis Context:
                {character_context}
                
                Narrative Analysis Context:
                {narrative_context}
                
                Provide detailed emotional impact analysis in JSON format including:
                - Emotional arc mapping
                - Scene-by-scene emotional beats
                - Character emotional journeys
                - Reader impact predictions
                - Enhancement recommendations
                """
            )

            # Create and run the chain
            chain = prompt | self.llm | StrOutputParser()
            
            analysis = await chain.ainvoke({
                "manuscript": manuscript_state.manuscript,
                "title": manuscript_state.title,
                "character_context": state.characters.content if state.characters else {},
                "narrative_context": state.story_arc.content if state.story_arc else {}
            })

            return AgentOutput(
                content=analysis,
                timestamp=datetime.now(),
                agent_id="emotional_impact_optimizer"
            ).dict()

        except Exception as e:
            logger.error(f"Emotional impact analysis failed: {str(e)}")
            return self.handle_error(e)
