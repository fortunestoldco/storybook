from typing import Dict, Any, List, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from .states import *
from .prompts import *
from .configuration import Configuration

def create_research_subgraph(research_type: str, state_class: Type[ReportState], config: Configuration) -> StateGraph:
    """Create a research subgraph based on the RESEARCHER-TO-INTEGRATE pattern."""
    
    # Create the section research subgraph
    section_builder = StateGraph(state_class)
    
    # Add nodes from RESEARCHER-TO-INTEGRATE
    section_builder.add_node("generate_queries", generate_queries)
    section_builder.add_node("search_web", search_web)
    section_builder.add_node("write_section", write_section)
    
    # Add edges
    section_builder.add_edge(START, "generate_queries")
    section_builder.add_edge("generate_queries", "search_web")
    section_builder.add_edge("search_web", "write_section")
    
    # Create outer research graph
    builder = StateGraph(state_class)
    
    # Add nodes from RESEARCHER-TO-INTEGRATE
    builder.add_node("generate_research_plan", generate_research_plan)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("build_section_with_web_research", section_builder.compile())
    builder.add_node("gather_findings", gather_findings)
    builder.add_node("write_final_report", write_final_report)
    builder.add_node("compile_report", compile_report)
    
    # Add edges
    builder.add_edge(START, "generate_research_plan")
    builder.add_edge("generate_research_plan", "human_feedback")
    builder.add_edge("conduct_research", "gather_findings")
    builder.add_conditional_edges("gather_findings", initiate_final_report, ["write_final_report"])
    builder.add_edge("write_final_report", "compile_report")
    builder.add_edge("compile_report", END)

    return builder.compile()
