from typing import Dict, Any, List, Literal, Type
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from ..utils.vector_search import VectorSearch
from .states import *
from .prompts import *
from .tools import *
from .configuration import Configuration
from .search import select_and_execute_search
from ..utils import load_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver

def create_research_subgraph(research_type: str, state_class: Type[ResearchState], config: Configuration) -> StateGraph:
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
    
    # Add nodes for research workflow
    async def generate_research_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate an initial research plan based on the query context."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Generate search queries
        writer_model = load_chat_model("research_query_generator", configuration)
        
        # Format system instructions based on research type
        if research_type == "domain":
            instructions = domain_research_instructions
        elif research_type == "cultural":
            instructions = cultural_research_instructions
        elif research_type == "market": 
            instructions = market_research_instructions
        elif research_type == "fact":
            instructions = fact_verification_instructions
        else:
            instructions = general_research_instructions
            
        # Generate queries for initial research
        query_messages = [
            SystemMessage(content=instructions),
            HumanMessage(content=f"Generate {configuration.queries_per_iteration} research queries for: {state.query_context}")
        ]
        
        query_results = await writer_model.ainvoke(query_messages)
        
        # Process queries
        queries = []
        for q in query_results.queries:
            queries.append(ResearchQuery(
                query=q,
                context=state.query_context,
                topic=state.topic,
                depth="standard"
            ))
        
        return {"queries": queries, "status": "plan_generated"}
    
    async def conduct_research(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Conduct research by executing queries."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        search_api = configuration.search_api
        search_api_config = configuration.search_api_config or {}
        
        # Execute searches
        query_list = [q.query for q in state.queries]
        search_results = await select_and_execute_search(
            search_api=search_api,
            query_list=query_list,
            params_to_pass=search_api_config
        )
        
        # Process search results
        processed_results = []
        sources = []
        
        for result in search_results:
            if not result["results"]:
                continue
                
            for item in result["results"]:
                processed_results.append(ResearchResult(
                    source_title=item.get("title", "Unknown Source"),
                    source_url=item.get("url", ""),
                    content=item.get("content", ""),
                    relevance_score=item.get("score", 0.5)
                ))
                
                sources.append(item.get("url", ""))
        
        return {
            "research_results": processed_results, 
            "sources": list(set(sources)),
            "status": "research_conducted"
        }
    
    async def analyze_findings(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze and synthesize research findings."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Load analysis model
        analysis_model = load_chat_model("research_quality_analyzer", configuration)
        
        # Format research results as context
        research_context = "\n\n".join([
            f"Source: {r.source_title}\nURL: {r.source_url}\n{r.content}"
            for r in state.research_results
        ])
        
        # Format analysis prompt based on research type
        if research_type == "domain":
            instructions = domain_synthesis_instructions
        elif research_type == "cultural":
            instructions = cultural_synthesis_instructions
        elif research_type == "market":
            instructions = market_synthesis_instructions
        elif research_type == "fact":
            instructions = fact_verification_synthesis_instructions
        else:
            instructions = general_synthesis_instructions
        
        # Generate synthesis
        synthesis_messages = [
            SystemMessage(content=instructions),
            HumanMessage(content=f"Topic: {state.topic}\n\nResearch Results:\n{research_context}")
        ]
        
        synthesis_result = await analysis_model.ainvoke(synthesis_messages)
        
        # Create research report
        report = ResearchReport(
            project_id=state.project_id,
            agent_name=research_type + "_research_agent",
            topic=state.topic,
            query_context=state.query_context,
            findings=synthesis_result.findings,
            sources=state.sources,
            confidence_score=synthesis_result.confidence
        )
        
        return {
            "report": report,
            "status": "findings_analyzed"
        }
    
    async def evaluate_quality(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Evaluate the quality of research findings."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Execute quality assessment
        quality_assessment = await analyze_research_quality(
            state.report,
            {
                "model": configuration.agent_model_configs.get("research_quality_analyzer", {})
            }
        )
        
        return {
            "quality_assessment": quality_assessment,
            "status": "quality_evaluated"
        }
    
    def determine_next_steps(state: ResearchState) -> Literal["identify_gaps", "compile_report"]:
        """Determine whether to continue research or finalize the report."""
        # Check if we've reached max iterations
        if state.iterations >= state.max_iterations:
            return "compile_report"
            
        # Check if quality is sufficient
        quality_score = state.quality_assessment.get("score", 0)
        if quality_score >= state.quality_threshold:
            return "compile_report"
            
        return "identify_gaps"
    
    async def identify_gaps(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Identify gaps in current research."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Identify knowledge gaps
        gaps = await identify_knowledge_gaps(state.report, {
            "model": configuration.agent_model_configs.get("research_gap_analyzer", {})
        })
        
        return {
            "identified_gaps": gaps,
            "status": "gaps_identified"
        }
    
    async def generate_followup_queries(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate follow-up queries based on identified gaps."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Generate follow-up queries
        followup_queries = await generate_followup_queries(
            state.identified_gaps,
            state.query_context,
            {
                "model": configuration.agent_model_configs.get("research_query_generator", {}),
                "queries_per_iteration": configuration.queries_per_iteration
            }
        )
        
        return {
            "queries": followup_queries,
            "iterations": state.iterations + 1,
            "status": "queries_generated"
        }
    
    async def compile_report(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Compile the final research report."""
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Store report in database if MongoDB is configured
        if configuration.mongodb_connection_string:
            from .storage import ResearchStorage
            storage = ResearchStorage()
            await storage.store_report(state.report)
        
        return {
            "final_report": state.report,
            "status": "completed"
        }
    
    # Add nodes to the graph
    builder.add_node("generate_research_plan", generate_research_plan)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("analyze_findings", analyze_findings)
    builder.add_node("evaluate_quality", evaluate_quality)
    builder.add_node("identify_gaps", identify_gaps)
    builder.add_node("generate_followup_queries", generate_followup_queries)
    builder.add_node("compile_report", compile_report)
    
    # Define the edges of the graph
    builder.add_edge(START, "generate_research_plan")
    builder.add_edge("generate_research_plan", "conduct_research")
    builder.add_edge("conduct_research", "analyze_findings")
    builder.add_edge("analyze_findings", "evaluate_quality")
    builder.add_conditional_edges(
        "evaluate_quality",
        determine_next_steps,
        {
            "identify_gaps": "identify_gaps",
            "compile_report": "compile_report"
        }
    )
    builder.add_edge("identify_gaps", "generate_followup_queries")
    builder.add_edge("generate_followup_queries", "conduct_research")
    builder.add_edge("compile_report", END)
    
    # Set up MongoDB checkpointing if configured
    if config.mongodb_connection_string and config.mongodb_database_name:
        checkpointer = MongoDBSaver(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database_name,
            collection_name=f"checkpoint_{research_type}_{state.project_id}"
        )
        graph = builder.compile(checkpointer=checkpointer)
    else:
        # Fallback to memory checkpointing
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
    
    return graph
