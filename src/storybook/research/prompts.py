# Copy all prompts from RESEARCHER-TO-INTEGRATE/prompts.py
report_planner_query_writer_instructions = """You are performing research...
# ... rest of prompts ...
"""

# Add our specialized prompts
domain_research_instructions = """You are analyzing domain-specific knowledge...
# ... additional specialized prompts ...
the t"""

# Copy quality analysis prompts from RESEARCHER-TO-INTEGRATE
quality_analysis_instructions = """Analyze the quality and completeness of research findings.

<Format>
{
    "score": float,  # Quality score between 0-1
    "feedback": List[str]  # List of specific feedback points
}
</Format>
"""

gap_analysis_instructions = """Identify gaps in the current research findings.

<Format>
{
    "gaps": List[str]  # List of identified knowledge gaps
}
</Format>
"""

followup_query_instructions = """Generate targeted search queries to fill knowledge gaps.

<Format>
{
    "queries": List[Dict] = [
        {
            "query": str,  # The search query
            "topic": str,  # Specific topic being researched
            "depth": str = "standard"  # Research depth (standard/deep)
        }
    ]
}
</Format>
"""

# ... rest of specialized prompts ...