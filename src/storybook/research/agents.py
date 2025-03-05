import os
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from ..agents.base import BaseAgent
from ..models.system import NovelSystemState
from ..storage.research import ResearchStorage
from .tools import (
    execute_research,
    analyze_research_quality,
    identify_knowledge_gaps,
    generate_followup_queries
)
from .states import (
    ResearchState,
    ResearchReport,
    ResearchIteration,
    ResearchQuery
)
from .config import validate_api_configuration, get_api_key
from .graphs import create_research_subgraph

class ResearchAgent(BaseAgent):
    """Base class for research-focused agents."""
    
    def __init__(self, name: str, research_graph: StateGraph, state_class: Type[ResearchState], config: Dict[str, Any]):
        """Initialize a research agent.
        
        Args:
            name: Agent name
            research_graph: Research subgraph for this agent
            state_class: State class for research operations
            config: Configuration for research operations
        """
        super().__init__(name, [], "", "")
        
        if not validate_api_configuration(config):
            raise ValueError(f"Invalid API configuration for {name}")
            
        self.research_graph = research_graph
        self.state_class = state_class
        self.research_config = config
        self.research_config["api_key"] = get_api_key(config["search_api"])
        self.storage = ResearchStorage()
        
        # Configure from env defaults if not specified
        self.max_iterations = config.get("max_iterations", 
            int(os.getenv("DEFAULT_MAX_RESEARCH_ITERATIONS", 3)))
        self.queries_per_iteration = config.get("queries_per_iteration",
            int(os.getenv("DEFAULT_QUERIES_PER_ITERATION", 3)))
        self.quality_threshold = config.get("quality_threshold",
            float(os.getenv("DEFAULT_QUALITY_THRESHOLD", 0.8)))
        self.cache_results = config.get("cache_results",
            os.getenv("ENABLE_RESEARCH_CACHE", "true").lower() == "true")
    
    async def process(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Process the current task through research.
        
        Args:
            state: Current system state
            config: Runtime configuration
            
        Returns:
            Updated state with research results
        """
        # Initialize research state
        research_state = self.state_class(
            project_id=state.project_id,
            topic=state.current_input.get("task"),
            query_context=state.project.synopsis,
            queries=[
                ResearchQuery(
                    query=state.current_input.get("task", ""),
                    context=state.project.synopsis,
                    topic=state.current_input.get("task", ""),
                    depth="standard"
                )
            ],
            iterations=0,
            max_iterations=self.max_iterations,
            quality_threshold=self.quality_threshold
        )
        
        # Execute research graph
        result = await self.research_graph.ainvoke(research_state, config)
        
        # Extract report from results
        report = result.get("final_report")
        if not report:
            return {
                "messages": [
                    AIMessage(content=f"Unable to complete research on {state.current_input.get('task')}. Please try again with a more specific query.")
                ]
            }
        
        # Format research findings as a response
        findings = "\n\n".join(report.findings)
        source_list = "\n".join([f"- {source}" for source in report.sources])
        
        response = f"""# Research Findings: {report.topic}

{findings}

## Sources
{source_list}

*Confidence Score: {report.confidence_score:.2f}*
"""
        
        # Return response
        return {
            "messages": [
                AIMessage(content=response)
            ],
            "agent_outputs": {
                **state.agent_outputs,
                self.name: state.agent_outputs.get(self.name, []) + [{
                    "timestamp": datetime.now().isoformat(),
                    "task": state.current_input.get("task", ""),
                    "response": response,
                    "report_id": report.report_id
                }]
            }
        }

class DomainKnowledgeSpecialist(ResearchAgent):
    """Specializes in technical and domain-specific research."""
    
    async def __call__(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute domain knowledge research.
        
        Args:
            state: Current system state
            config: Runtime configuration
            
        Returns:
            Updated state with research results
        """
        result = await self.process(state, config)
        
        # Enhance domain-specific results with additional context
        if result.get("messages"):
            content = result["messages"][0].content
            enhanced_content = f"""# Domain Knowledge Research

{content}

## Domain-Specific Applications
- Consider how this knowledge affects your characters' behaviors and dialogue
- Integrate domain-specific terminology naturally through expert characters
- Use these concepts to create authentic challenges and plot developments

"""
            result["messages"][0] = AIMessage(content=enhanced_content)
            
        return result

class CulturalAuthenticityExpert(ResearchAgent):
    """Specializes in cultural and sociological research."""
    
    async def __call__(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute cultural research.
        
        Args:
            state: Current system state
            config: Runtime configuration
            
        Returns:
            Updated state with research results
        """
        result = await self.process(state, config)
        
        # Enhance cultural research with additional context
        if result.get("messages"):
            content = result["messages"][0].content
            enhanced_content = f"""# Cultural Research

{content}

## Cultural Authenticity Guidelines
- Avoid stereotyping or generalizing cultural practices
- Consider diverse perspectives within the culture
- Be mindful of cultural contexts and historical factors
- Consult sensitivity readers for final validation

"""
            result["messages"][0] = AIMessage(content=enhanced_content)
            
        return result

class MarketAlignmentDirector(ResearchAgent):
    """Specializes in market research and trend analysis."""
    
    async def __call__(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute market research.
        
        Args:
            state: Current system state
            config: Runtime configuration
            
        Returns:
            Updated state with research results
        """
        result = await self.process(state, config)
        
        # Enhance market research with additional context
        if result.get("messages"):
            content = result["messages"][0].content
            enhanced_content = f"""# Market Research

{content}

## Market Positioning Recommendations
- Consider how your novel fits within current market trends
- Identify unique selling points to differentiate your work
- Target specific reader demographics based on content and themes
- Develop promotional strategies aligned with market expectations

"""
            result["messages"][0] = AIMessage(content=enhanced_content)
            
        return result

class FactVerificationSpecialist(ResearchAgent):
    """Specializes in fact checking and verification."""
    
    async def __call__(self, state: NovelSystemState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute fact verification research.
        
        Args:
            state: Current system state
            config: Runtime configuration
            
        Returns:
            Updated state with research results
        """
        result = await self.process(state, config)
        
        # Enhance fact verification with additional context
        if result.get("messages"):
            content = result["messages"][0].content
            enhanced_content = f"""# Fact Verification

{content}

## Accuracy Guidelines
- When uncertain, either research further or avoid definitive statements
- Consider including an author's note for creative liberties taken
- Balance factual accuracy with narrative requirements
- Document sources for future reference and verification

"""
            result["messages"][0] = AIMessage(content=enhanced_content)
            
        return result