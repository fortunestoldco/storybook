from typing import List, Dict, Any, Optional, Union
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from ..utils import load_chat_model

from .states import ResearchQuery, ResearchResult, ResearchReport
from .search import select_and_execute_search
from .prompts import (
    quality_analysis_instructions,
    gap_analysis_instructions,
    followup_query_instructions
)

@tool
async def execute_research(queries: List[ResearchQuery], config: Dict[str, Any]) -> List[ResearchResult]:
    """Execute research queries across configured search APIs."""
    search_api = config["search_api"]
    search_params = config.get("search_api_config", {})
    
    results = await select_and_execute_search(
        search_api=search_api,
        queries=[q.query for q in queries],
        params=search_params
    )
    
    return [
        ResearchResult(
            source_title=r["title"],
            source_url=r["url"], 
            content=r["content"],
            relevance_score=r.get("score", 0.5)
        )
        for r in results
    ]

@tool
def analyze_research_quality(report: ResearchReport, config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze quality and completeness of research."""
    analysis_llm = load_chat_model("research_quality_analyzer", config)
    
    analysis = analysis_llm.invoke([
        SystemMessage(content=quality_analysis_instructions),
        HumanMessage(content=str({
            "findings": report.findings,
            "sources": report.sources,
            "topic": report.topic
        }))
    ])
    
    return {
        "score": analysis.get("score", 0.0),
        "feedback": analysis.get("feedback", [])
    }

@tool 
def identify_knowledge_gaps(report: ResearchReport, config: Dict[str, Any]) -> List[str]:
    """Identify gaps in current research findings."""
    analysis_llm = load_chat_model("research_gap_analyzer", config)
    
    analysis = analysis_llm.invoke([
        SystemMessage(content=gap_analysis_instructions),
        HumanMessage(content=str({
            "findings": report.findings,
            "topic": report.topic,
            "context": report.query_context
        }))
    ])
    
    return analysis.get("gaps", [])

@tool
def generate_followup_queries(gaps: List[str], context: str, config: Dict[str, Any]) -> List[ResearchQuery]:
    """Generate targeted queries to fill knowledge gaps."""
    query_llm = load_chat_model("research_query_generator", config)
    
    queries = query_llm.invoke([
        SystemMessage(content=followup_query_instructions),
        HumanMessage(content=str({
            "gaps": gaps,
            "context": context
        }))
    ])
    
    return [
        ResearchQuery(
            query=q["query"],
            context=context,
            topic=q["topic"],
            depth=q.get("depth", "standard")
        )
        for q in queries.get("queries", [])
    ]