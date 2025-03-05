from typing import Dict, Any, List, Literal, Type, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from datetime import datetime

from .states import (
    ResearchState,
    DomainResearchState,
    CulturalResearchState,
    MarketResearchState,
    FactVerificationState
)
from .tools import (
    execute_research,
    analyze_research_quality,
    identify_knowledge_gaps,
    generate_followup_queries
)
from .configuration import Configuration

def create_research_subgraph(
    research_type: str, 
    state_class: Type[ResearchState], 
    config: Optional[Configuration] = None
) -> StateGraph:
    """Create a research subgraph for a specific type of research.
    
    Args:
        research_type: Type of research (domain, cultural, market, fact)
        state_class: State class for tracking research state
        config: System configuration
        
    Returns:
        StateGraph for research workflow
    """
    # Create state graph
    builder = StateGraph(state_class)
    
    # Define research configuration if not provided
    if config is None:
        config = Configuration()
    
    # Add nodes for the research workflow
    async def initialize_research(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Initialize the research process by setting up the state."""
        # Keep any existing state properties
        return {
            "status": "initialized",
            "iterations": 0
        }
    
    async def execute_search(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute search queries and update state with results."""
        # Execute research queries
        results = await execute_research(state.queries, {
            "search_api": config.get("configurable", {}).get("search_api", "tavily"),
            "search_api_config": config.get("configurable", {}).get("search_api_config", {})
        })
        
        # Create a simple report from results
        findings = [f"Finding from {r.source_title}: {r.content}" for r in results]
        sources = [r.source_url for r in results]
        
        # Create a research report
        report = {
            "agent_name": f"{research_type}_research_agent",
            "topic": state.queries[0].topic if state.queries else "Unknown",
            "query_context": state.query_context,
            "findings": findings,
            "sources": sources,
            "confidence_score": 0.7  # Placeholder confidence score
        }
        
        return {
            "results": results,
            "report": report,
            "status": "search_completed"
        }
    
    async def analyze_quality(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze the quality of research results."""
        quality_assessment = await analyze_research_quality(state.report, {
            "model": config.get("configurable", {}).get("agent_model_configs", {}).get("research_quality_analyzer", {})
        })
        
        return {
            "quality_assessment": quality_assessment,
            "status": "quality_analyzed"
        }
    
    def determine_next_steps(state: ResearchState) -> Literal["identify_gaps", "finalize_research"]:
        """Determine whether more research is needed or we can finalize."""
        # Check if max iterations reached
        if state.iterations >= state.metadata.get("max_iterations", 3):
            return "finalize_research"
        
        # Check if quality threshold met
        quality_score = state.quality_assessment.get("score", 0.0)
        if quality_score >= state.metadata.get("quality_threshold", 0.7):
            return "finalize_research"
        
        # Otherwise, continue with more research
        return "identify_gaps"
    
    async def identify_research_gaps(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Identify gaps in the research."""
        gaps = await identify_knowledge_gaps(state.report, {
            "model": config.get("configurable", {}).get("agent_model_configs", {}).get("research_gap_analyzer", {})
        })
        
        return {
            "gaps": gaps,
            "status": "gaps_identified"
        }
    
    async def generate_new_queries(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate new queries based on identified gaps."""
        followup_queries = await generate_followup_queries(
            state.gaps,
            state.query_context,
            {
                "model": config.get("configurable", {}).get("agent_model_configs", {}).get("research_query_generator", {}),
                "queries_per_iteration": config.get("configurable", {}).get("queries_per_iteration", 3)
            }
        )
        
        return {
            "queries": followup_queries,
            "iterations": state.iterations + 1,
            "status": "new_queries_generated"
        }
    
    async def finalize_research(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Finalize the research report."""
        # Could store to a database here in a full implementation
        
        return {
            "final_report": state.report,
            "status": "completed"
        }
    
    # Add nodes to the graph
    builder.add_node("initialize_research", initialize_research)
    builder.add_node("execute_search", execute_search)
    builder.add_node("analyze_quality", analyze_quality)
    builder.add_node("identify_gaps", identify_research_gaps)
    builder.add_node("generate_new_queries", generate_new_queries)
    builder.add_node("finalize_research", finalize_research)
    
    # Define the edges of the graph
    builder.add_edge(START, "initialize_research")
    builder.add_edge("initialize_research", "execute_search")
    builder.add_edge("execute_search", "analyze_quality")
    builder.add_conditional_edges(
        "analyze_quality",
        determine_next_steps,
        {
            "identify_gaps": "identify_gaps",
            "finalize_research": "finalize_research"
        }
    )
    builder.add_edge("identify_gaps", "generate_new_queries")
    builder.add_edge("generate_new_queries", "execute_search")
    builder.add_edge("finalize_research", END)
    
    # Compile and return the graph
    return builder.compile()