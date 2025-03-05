"""Prompts for research operations."""

# Quality analysis prompts
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