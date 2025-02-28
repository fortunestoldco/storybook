from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional
import logging
import json
import re

# Third-party imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document  # Already correctly imported

# Local imports
from storybook.config import get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class QualityReviewer:
    """Agent responsible for final quality review and manuscript finalization."""

    def __init__(self):
        self.llm = get_llm(temperature=0.5, use_replicate=True)
        self.document_store = DocumentStore()

    def finalize_manuscript(
        self,
        manuscript_id: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform final quality review and finalize the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}

        # Perform quality review
        quality_review = self._review_quality(
            manuscript["content"], target_audience, research_insights
        )

        # Identify remaining improvement opportunities
        improvement_opportunities = self._identify_improvement_opportunities(
            manuscript["content"], quality_review, target_audience
        )

        # Generate final report
        final_report = self._generate_final_report(
            manuscript_id,
            manuscript["title"],
            quality_review,
            improvement_opportunities,
            target_audience,
            research_insights,
        )

        # Optionally make final tweaks
        if improvement_opportunities.get("critical_issues", []):
            updated_content = self._make_final_improvements(
                manuscript["content"], improvement_opportunities, target_audience
            )

            # Store the updated manuscript
            self.document_store.update_manuscript(
                manuscript_id, {"content": updated_content}
            )

        return {
            "manuscript_id": manuscript_id,
            "message": "Completed final quality review and manuscript finalization.",
            "review": quality_review,
            "improvements": improvement_opportunities,
            "final_report": final_report,
        }

    def _review_quality(
        self,
        content: str,
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive quality review of the manuscript."""
        # Sample representative sections for review
        sample_size = min(8000, len(content) // 3)
        beginning = content[:sample_size]
        middle_start = max(0, (len(content) // 2) - (sample_size // 2))
        middle = content[middle_start : middle_start + sample_size]
        end_start = max(0, len(content) - sample_size)
        end = content[end_start:]

        # Combine samples
        sample = f"BEGINNING:\n{beginning}\n\nMIDDLE:\n{middle}\n\nEND:\n{end}"

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Reading Preferences: {target_audience.get('reading_preferences', {}).get('reading', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            
            Evaluate how well the manuscript meets the expectations of this target audience.
            """

        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            
            Consider how the manuscript aligns with current market trends and reader preferences.
            """

        # Create prompt for quality review
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Senior Editor performing a comprehensive quality review of a manuscript.
        
        Manuscript Samples:
        {manuscript_samples}
        
        {audience_context}
        
        {research_context}
        
        Evaluate the following aspects:
        
        1. Overall Narrative Quality: How compelling and engaging is the story?
        2. Character Development: How well-developed and believable are the characters?
        3. World-Building: How rich and immersive is the setting?
        4. Plot Coherence: How logical and well-structured is the plot?
        5. Pacing: How appropriate is the story's rhythm and flow?
        6. Dialogue: How authentic and purposeful is the dialogue?
        7. Prose Quality: How polished and effective is the writing?
        8. Thematic Depth: How effectively are themes explored?
        9. Market Readiness: How ready is this manuscript for its target market?
        10. Reader Engagement: How likely is this manuscript to captivate readers?
        
        For each aspect, provide:
        - A rating from 1-10
        - Specific strengths
        - Specific weaknesses
        - Brief examples from the text
        
        Conclude with an overall assessment and recommendation.
        """
        )

        # Create the chain
        chain = (
            {
                "manuscript_samples": lambda _: sample,
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        review = chain.invoke("Perform quality review")

        # Parse the review to extract ratings
        import re

        # Extract ratings
        ratings = {}

        aspects = [
            "Overall Narrative Quality",
            "Character Development",
            "World-Building",
            "Plot Coherence",
            "Pacing",
            "Dialogue",
            "Prose Quality",
            "Thematic Depth",
            "Market Readiness",
            "Reader Engagement",
        ]

        for aspect in aspects:
            pattern = rf"{aspect}:.*?(\d+)/10"
            match = re.search(pattern, review, re.IGNORECASE | re.DOTALL)
            if match:
                ratings[aspect.lower().replace(" ", "_")] = int(match.group(1))

        # Extract overall assessment
        assessment_match = re.search(
            r"Overall(?: Assessment)?:?(.*?)(?=\n\n|\Z)",
            review,
            re.IGNORECASE | re.DOTALL,
        )
        overall_assessment = (
            assessment_match.group(1).strip() if assessment_match else ""
        )

        # Calculate average rating
        if ratings:
            avg_rating = sum(ratings.values()) / len(ratings)
        else:
            avg_rating = 0

        return {
            "full_review": review,
            "ratings": ratings,
            "overall_assessment": overall_assessment,
            "average_rating": avg_rating,
        }

    def _identify_improvement_opportunities(
        self,
        content: str,
        quality_review: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Identify remaining improvement opportunities based on the quality review."""
        # Extract ratings from quality review
        ratings = quality_review.get("ratings", {})

        # Find aspects with the lowest ratings
        sorted_aspects = sorted(ratings.items(), key=lambda x: x[1])
        weakest_aspects = (
            sorted_aspects[:3] if len(sorted_aspects) >= 3 else sorted_aspects
        )

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Focus on improvements that would make the manuscript more appealing to this audience.
            """

        # Create prompt for identifying improvement opportunities
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Manuscript Improvement Specialist. Based on the quality review, identify the most important
        remaining improvements needed for this manuscript.
        
        Quality Review:
        {quality_review}
        
        Weakest Areas:
        {weakest_areas}
        
        {audience_context}
        
        Identify:
        
        1. Critical Issues: 2-3 serious problems that could significantly impact reader experience
        2. Quick Wins: 3-5 easily addressable issues that would yield noticeable improvements
        3. Long-term Revisions: 2-3 deeper issues that would require more extensive revision
        
        For each issue:
        - Clearly describe the problem
        - Explain its impact on the reader
        - Provide a specific recommendation for addressing it
        - If possible, suggest where in the manuscript this issue is most evident
        
        Focus on concrete, actionable improvements rather than generalities.
        """
        )

        # Format weakest areas
        weakest_areas_text = ""
        for aspect, rating in weakest_aspects:
            weakest_areas_text += f"{aspect.replace('_', ' ').title()}: {rating}/10\n"

        # Create the chain
        chain = (
            {
                "quality_review": lambda _: quality_review.get("full_review", ""),
                "weakest_areas": lambda _: weakest_areas_text,
                "audience_context": lambda _: audience_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        opportunities = chain.invoke("Identify improvement opportunities")

        # Parse the opportunities into sections
        import re

        # Extract critical issues
        critical_match = re.search(
            r"Critical Issues:?(.*?)(?=\n\n\d+\.|Quick Wins:|Long-term Revisions:|\Z)",
            opportunities,
            re.IGNORECASE | re.DOTALL,
        )
        critical_issues = critical_match.group(1).strip() if critical_match else ""

        # Extract quick wins
        quick_match = re.search(
            r"Quick Wins:?(.*?)(?=\n\n\d+\.|Critical Issues:|Long-term Revisions:|\Z)",
            opportunities,
            re.IGNORECASE | re.DOTALL,
        )
        quick_wins = quick_match.group(1).strip() if quick_match else ""

        # Extract long-term revisions
        longterm_match = re.search(
            r"Long-term Revisions:?(.*?)(?=\n\n\d+\.|Critical Issues:|Quick Wins:|\Z)",
            opportunities,
            re.IGNORECASE | re.DOTALL,
        )
        longterm_revisions = longterm_match.group(1).strip() if longterm_match else ""

        # Try to extract individual issues
        def extract_issues(text):
            issues = []
            # Look for numbered issues (e.g., "1. Issue description")
            issue_blocks = re.findall(r"(\d+\.\s+.*?)(?=\n\n\d+\.|\Z)", text, re.DOTALL)
            if not issue_blocks:
                # Try another pattern (e.g., "Issue 1: description")
                issue_blocks = re.findall(
                    r"((?:Issue\s+)?\d+:?\s+.*?)(?=\n\n(?:Issue\s+)?\d+:?|\Z)",
                    text,
                    re.DOTALL,
                )

            for block in issue_blocks:
                issues.append(block.strip())

            return issues

        critical_issues_list = extract_issues(critical_issues)
        quick_wins_list = extract_issues(quick_wins)
        longterm_revisions_list = extract_issues(longterm_revisions)

        return {
            "full_analysis": opportunities,
            "critical_issues": critical_issues_list,
            "quick_wins": quick_wins_list,
            "longterm_revisions": longterm_revisions_list,
        }

    def _generate_final_report(
        self,
        manuscript_id: str,
        manuscript_title: str,
        quality_review: Dict[str, Any],
        improvement_opportunities: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
        research_insights: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate comprehensive final report on the manuscript."""
        # Format quality ratings
        ratings = quality_review.get("ratings", {})
        ratings_text = ""
        for aspect, rating in ratings.items():
            ratings_text += f"{aspect.replace('_', ' ').title()}: {rating}/10\n"

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience Information:
            - Demographic: {target_audience.get('demographic', 'General readers')}
            - Reading Preferences: {target_audience.get('reading_preferences', {}).get('reading', 'Various preferences')}
            - Content Expectations: {target_audience.get('content_expectations', {}).get('values', 'Various values')}
            """

        # Add research context if available
        research_context = ""
        if research_insights:
            research_context = f"""
            Market Research Insights:
            {research_insights.get('market_analysis_summary', 'No market analysis available')}
            """

        # Create prompt for final report
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Senior Editor preparing a comprehensive final report on a manuscript.
        
        Manuscript Title: {manuscript_title}
        
        Quality Ratings:
        {quality_ratings}
        
        Overall Assessment:
        {overall_assessment}
        
        Improvement Opportunities:
        Critical Issues: {critical_issues_count}
        Quick Wins: {quick_wins_count}
        Long-term Revisions: {longterm_revisions_count}
        
        {audience_context}
        
        {research_context}
        
        Write a comprehensive, professional final report covering:
        
        1. Executive Summary: Brief overview of the manuscript's current state
        2. Strengths: The manuscript's strongest elements
        3. Areas for Improvement: Most critical issues to address
        4. Market Potential: How the manuscript might perform in the current market
        5. Target Audience Alignment: How well it meets target audience expectations
        6. Recommended Next Steps: Clear action items for the author
        7. Publication Readiness: Assessment of how close the manuscript is to being ready for publication
        
        The report should be thorough but concise, providing actionable insights that will help 
        the author make final decisions about the manuscript.
        """
        )

        # Create the chain
        chain = (
            {
                "manuscript_title": lambda _: manuscript_title,
                "quality_ratings": lambda _: ratings_text,
                "overall_assessment": lambda _: quality_review.get(
                    "overall_assessment", ""
                ),
                "critical_issues_count": lambda _: len(
                    improvement_opportunities.get("critical_issues", [])
                ),
                "quick_wins_count": lambda _: len(
                    improvement_opportunities.get("quick_wins", [])
                ),
                "longterm_revisions_count": lambda _: len(
                    improvement_opportunities.get("longterm_revisions", [])
                ),
                "audience_context": lambda _: audience_context,
                "research_context": lambda _: research_context,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        final_report = chain.invoke("Generate final report")

        # Store the report in the database
        self.document_store.store_analysis_document(
            manuscript_id,
            "final_quality_report",
            {
                "title": f"Final Quality Report for '{manuscript_title}'",
                "report": final_report,
                "quality_review": quality_review,
                "improvement_opportunities": improvement_opportunities,
            },
        )

        return final_report

    def _make_final_improvements(
        self,
        content: str,
        improvement_opportunities: Dict[str, Any],
        target_audience: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make final critical improvements to the manuscript."""
        # Extract critical issues
        critical_issues = improvement_opportunities.get("critical_issues", [])

        # If no critical issues, return content unchanged
        if not critical_issues:
            return content

        # We'll focus on addressing 1-2 critical issues
        issues_to_address = critical_issues[: min(2, len(critical_issues))]

        # Add audience context if available
        audience_context = ""
        if target_audience:
            audience_context = f"""
            Target Audience: {target_audience.get('demographic', 'General readers')}
            Ensure improvements will appeal to this audience.
            """

        # Create prompt for final improvements
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Manuscript Finalization Specialist. Address this critical issue in the manuscript.
        
        Critical Issue:
        {critical_issue}
        
        {audience_context}
        
        To address this issue, you'll need to enhance the manuscript. Create a small section (2-3 paragraphs)
        that could be inserted into the manuscript to address this issue. This section should:
        
        1. Directly address the critical issue described
        2. Match the style and tone of the manuscript
        3. Integrate seamlessly with the existing content
        4. Add significant value to the reader experience
        
        Write this section as if it were part of the manuscript itself, ready to be inserted.
        """
        )

        # Apply improvements
        updated_content = content

        for issue in issues_to_address:
            # Create the chain
            chain = (
                {
                    "critical_issue": lambda _: issue,
                    "audience_context": lambda _: audience_context,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            improvement = chain.invoke(f"Address critical issue: {issue[:50]}...")

            # Find a suitable insertion point
            # For simplicity, we'll insert near the beginning, middle, or end
            # based on the issue description
            issue_lower = issue.lower()

            insertion_point = len(updated_content) // 2  # Default to middle

            if any(
                term in issue_lower
                for term in ["beginning", "start", "opening", "introduction"]
            ):
                # Insert near beginning (after first 10% of content)
                insertion_point = len(updated_content) // 10
            elif any(
                term in issue_lower
                for term in ["end", "conclusion", "resolution", "final"]
            ):
                # Insert near end (before last 10% of content)
                insertion_point = int(len(updated_content) * 0.9)

            # Find the nearest paragraph break
            paragraph_break = updated_content.rfind("\n\n", 0, insertion_point)
            if paragraph_break == -1:
                paragraph_break = updated_content.find("\n\n", insertion_point)

            if paragraph_break != -1:
                # Insert the improvement
                updated_content = (
                    updated_content[:paragraph_break]
                    + "\n\n"
                    + improvement
                    + "\n\n"
                    + updated_content[paragraph_break:]
                )

        return updated_content
