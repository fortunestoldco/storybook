from typing import List, Dict, Any, Optional, Union
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from ..utils.chat_model import load_chat_model

from .states import ResearchQuery, ResearchResult, ResearchReport
from .search import select_and_execute_search
from .prompts import (
    quality_analysis_instructions,
    gap_analysis_instructions,
    followup_query_instructions
)

@tool
async def execute_research(queries: List[ResearchQuery], config: Dict[str, Any]) -> List[ResearchResult]:
    """Execute research queries across configured search APIs.
    
    Args:
        queries: List of research queries to execute
        config: Configuration for search execution
        
    Returns:
        List of research results
    """
    search_api = config.get("search_api", "tavily")
    search_params = config.get("search_api_config", {})
    
    results = await select_and_execute_search(
        search_api=search_api,
        query_list=[q.query for q in queries],
        params_to_pass=search_params
    )
    
    processed_results = []
    for r in results:
        if not r.get("results"):
            continue
            
        for item in r.get("results", []):
            processed_results.append(
                ResearchResult(
                    source_title=item.get("title", "Unknown Source"),
                    source_url=item.get("url", ""),
                    content=item.get("content", ""),
                    relevance_score=item.get("score", 0.5)
                )
            )
    
    return processed_results

@tool
async def analyze_research_quality(report: ResearchReport, config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze quality and completeness of research.
    
    Args:
        report: Research report to analyze
        config: Configuration for analysis
        
    Returns:
        Quality assessment with score and feedback
    """
    analysis_llm = load_chat_model("research_quality_analyzer", config)
    
    analysis_result = await analysis_llm.ainvoke([
        SystemMessage(content=quality_analysis_instructions),
        HumanMessage(content=str({
            "findings": report.findings,
            "sources": report.sources,
            "topic": report.topic
        }))
    ])
    
    # Extract score and feedback from structured output
    content = analysis_result.content
    if isinstance(content, str):
        # Try to parse JSON response
        try:
            import json
            parsed = json.loads(content)
            return {
                "score": parsed.get("score", 0.0),
                "feedback": parsed.get("feedback", [])
            }
        except:
            # Fallback to simple extraction
            import re
            score_match = re.search(r'"score":\s*([\d.]+)', content)
            score = float(score_match.group(1)) if score_match else 0.0
            
            feedback_match = re.search(r'"feedback":\s*\[(.*?)\]', content, re.DOTALL)
            feedback_str = feedback_match.group(1) if feedback_match else ""
            feedback = [f.strip(' "\'') for f in feedback_str.split(",")]
            
            return {
                "score": score,
                "feedback": feedback
            }
    
    # Return default if parsing fails
    return {
        "score": 0.0,
        "feedback": ["Unable to analyze report quality"]
    }

@tool 
async def identify_knowledge_gaps(report: ResearchReport, config: Dict[str, Any]) -> List[str]:
    """Identify gaps in current research findings.
    
    Args:
        report: Research report to analyze
        config: Configuration for gap analysis
        
    Returns:
        List of identified knowledge gaps
    """
    analysis_llm = load_chat_model("research_gap_analyzer", config)
    
    analysis_result = await analysis_llm.ainvoke([
        SystemMessage(content=gap_analysis_instructions),
        HumanMessage(content=str({
            "findings": report.findings,
            "topic": report.topic,
            "context": report.query_context
        }))
    ])
    
    # Extract gaps from structured output
    content = analysis_result.content
    if isinstance(content, str):
        # Try to parse JSON response
        try:
            import json
            parsed = json.loads(content)
            return parsed.get("gaps", [])
        except:
            # Fallback to simple extraction
            import re
            gaps_match = re.search(r'"gaps":\s*\[(.*?)\]', content, re.DOTALL)
            gaps_str = gaps_match.group(1) if gaps_match else ""
            gaps = [g.strip(' "\'') for g in gaps_str.split(",")]
            return gaps
    
    # Return default if parsing fails
    return ["More comprehensive information needed"]

@tool
async def generate_followup_queries(gaps: List[str], context: str, config: Dict[str, Any]) -> List[ResearchQuery]:
    """Generate targeted queries to fill knowledge gaps.
    
    Args:
        gaps: Identified knowledge gaps
        context: Original research context
        config: Configuration for query generation
        
    Returns:
        List of follow-up research queries
    """
    query_llm = load_chat_model("research_query_generator", config)
    
    query_result = await query_llm.ainvoke([
        SystemMessage(content=followup_query_instructions),
        HumanMessage(content=str({
            "gaps": gaps,
            "context": context
        }))
    ])
    
    # Extract queries from structured output
    content = query_result.content
    queries = []
    
    if isinstance(content, str):
        # Try to parse JSON response
        try:
            import json
            parsed = json.loads(content)
            query_list = parsed.get("queries", [])
            
            for q in query_list:
                queries.append(
                    ResearchQuery(
                        query=q.get("query", ""),
                        context=context,
                        topic=q.get("topic", ""),
                        depth=q.get("depth", "standard")
                    )
                )
        except:
            # Fallback to generating queries from gaps
            for gap in gaps:
                queries.append(
                    ResearchQuery(
                        query=f"Research about {gap}",
                        context=context,
                        topic=gap,
                        depth="standard"
                    )
                )
    
    # Ensure we have at least one query
    if not queries:
        queries.append(
            ResearchQuery(
                query=f"More information about {context}",
                context=context,
                topic=context,
                depth="standard"
            )
        )
    
    return queries