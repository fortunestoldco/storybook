from typing import Dict, List, Any, Optional
import logging
import json
import re

from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from storybook.config import get_llm
from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class ContinuityEditor:
    """Agent responsible for identifying and fixing continuity issues."""

    def __init__(self):
        self.llm = get_llm(
            temperature=0.3, use_replicate=True
        )  # Lower temperature for more consistent analysis
        self.document_store = DocumentStore()

    def check_and_fix_continuity(self, manuscript_id: str) -> Dict[str, Any]:
        """Identify and fix continuity issues in the manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return {"error": f"Manuscript {manuscript_id} not found"}

        # Identify continuity issues
        continuity_issues = self._identify_continuity_issues(manuscript["content"])

        # If no issues found, return early
        if not continuity_issues.get("issues", []):
            return {
                "manuscript_id": manuscript_id,
                "message": "No significant continuity issues identified.",
                "issues": [],
            }

        # Fix continuity issues
        updated_content, fixed_issues = self._fix_continuity_issues(
            manuscript["content"], continuity_issues
        )

        # Store the updated manuscript
        self.document_store.update_manuscript(
            manuscript_id, {"content": updated_content}
        )

        return {
            "manuscript_id": manuscript_id,
            "message": f"Identified and fixed {len(fixed_issues)} continuity issues.",
            "issues": fixed_issues,
            "original_issues": continuity_issues.get("issues", []),
        }

    def _identify_continuity_issues(self, content: str) -> Dict[str, Any]:
        """Identify continuity issues in the manuscript."""
        # Break the content into manageable chunks for analysis
        chunk_size = min(10000, len(content) // 3)
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            chunks.append(chunk)

        # Limit to a reasonable number of chunks
        chunks = chunks[:5]

        # Create prompt for continuity analysis
        prompt = ChatPromptTemplate.from_template(
            """
        You are a Continuity Editor specializing in identifying inconsistencies in manuscripts.
        
        Analyze the following manuscript excerpts for continuity issues such as:
        
        1. Character inconsistencies (names, descriptions, traits changing)
        2. Timeline inconsistencies (events out of order, impossible timing)
        3. Setting inconsistencies (locations changing unexpectedly)
        4. Plot inconsistencies (contradictory events or information)
        5. Logic inconsistencies (implausible or impossible situations)
        
        Manuscript Excerpts:
        {manuscript_excerpts}
        
        For each continuity issue you identify, provide:
        1. Issue Type: The category of continuity error
        2. Description: A clear explanation of the inconsistency
        3. Location: Where in the excerpt the issue appears (quote the relevant text)
        4. Severity: How significantly it impacts the story (High, Medium, Low)
        5. Recommended Fix: A specific suggestion to resolve the inconsistency
        
        Format your response as a JSON list of objects with the above keys.
        Focus only on clear continuity issues, not subjective writing preferences.
        """
        )

        # Format the chunks
        excerpts_text = []
        for i, chunk in enumerate(chunks):
            excerpts_text.append(
                f"EXCERPT {i+1} (approximately {i+1}/{len(chunks)} of the way through):\n{chunk}"
            )

        combined_excerpts = "\n\n" + "\n\n".join(excerpts_text)

        # Create the chain
        chain = (
            {"manuscript_excerpts": lambda _: combined_excerpts}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain
        issues_str = chain.invoke("Identify continuity issues")

        # Parse the issues
        try:
            issues_list = json.loads(issues_str)
            if isinstance(issues_list, list):
                return {"issues": issues_list}
            return {"issues": []}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse continuity issues JSON: {issues_str}")

            # Try to extract structured data manually
            issues = []

            # Try to find issue blocks
            issue_blocks = re.findall(
                r"(\d+\.\s+Issue Type:.*?)(?=\d+\.\s+Issue Type:|$)",
                issues_str,
                re.DOTALL,
            )

            for block in issue_blocks:
                issue = {}

                # Extract issue type
                type_match = re.search(r"Issue Type:?\s*(.*?)(?=\n|$)", block)
                if type_match:
                    issue["Issue Type"] = type_match.group(1).strip()

                # Extract description
                desc_match = re.search(
                    r"Description:?\s*(.*?)(?=\n\d|$)", block, re.DOTALL
                )
                if desc_match:
                    issue["Description"] = desc_match.group(1).strip()

                # Extract location
                loc_match = re.search(r"Location:?\s*(.*?)(?=\n\d|$)", block, re.DOTALL)
                if loc_match:
                    issue["Location"] = loc_match.group(1).strip()

                # Extract severity
                sev_match = re.search(r"Severity:?\s*(.*?)(?=\n|$)", block)
                if sev_match:
                    issue["Severity"] = sev_match.group(1).strip()

                # Extract recommended fix
                fix_match = re.search(
                    r"Recommended Fix:?\s*(.*?)(?=\n\d|$)", block, re.DOTALL
                )
                if fix_match:
                    issue["Recommended Fix"] = fix_match.group(1).strip()

                if issue:
                    issues.append(issue)

            return {"issues": issues}

    def _fix_continuity_issues(
        self, content: str, continuity_issues: Dict[str, Any]
    ) -> tuple:
        """Fix identified continuity issues in the manuscript."""
        issues = continuity_issues.get("issues", [])

        # Filter to only high and medium severity issues
        priority_issues = [
            issue
            for issue in issues
            if issue.get("Severity", "").lower() in ["high", "medium"]
        ]

        # If no priority issues, return content unchanged
        if not priority_issues:
            return content, []

        # Fix each issue one by one
        updated_content = content
        fixed_issues = []

        for issue in priority_issues:
            # Extract the issue information
            issue_type = issue.get("Issue Type", "")
            issue_location = issue.get("Location", "")
            recommended_fix = issue.get("Recommended Fix", "")

            # If we don't have a location or fix, skip
            if not issue_location or not recommended_fix:
                continue

            # Try to find the problematic text in the content
            # We'll look for a snippet from the location
            location_snippet = self._extract_snippet(issue_location)

            if not location_snippet or location_snippet not in updated_content:
                continue

            # Create prompt for fixing this specific issue
            prompt = ChatPromptTemplate.from_template(
                """
            You are a Continuity Editor fixing a specific inconsistency in a manuscript.
            
            Issue Type: {issue_type}
            Issue Description: {issue_description}
            
            Original Problematic Text:
            {problematic_text}
            
            Recommended Fix Approach:
            {recommended_fix}
            
            Provide a revised version of the problematic text that fixes the continuity issue.
            Make minimal changes - only what's necessary to resolve the inconsistency.
            Maintain the same style, tone, and length as the original text.
            
            Return ONLY the revised text that should directly replace the problematic section.
            """
            )

            # Create the chain
            chain = (
                {
                    "issue_type": lambda _: issue_type,
                    "issue_description": lambda _: issue.get("Description", ""),
                    "problematic_text": lambda _: self._get_context(
                        updated_content, location_snippet
                    ),
                    "recommended_fix": lambda _: recommended_fix,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            fixed_text = chain.invoke("Fix continuity issue")

            # Replace the problematic text with the fixed version
            context = self._get_context(updated_content, location_snippet)
            if context in updated_content:
                updated_content = updated_content.replace(context, fixed_text, 1)

                # Add to fixed issues
                fixed_issues.append(
                    {
                        "issue_type": issue_type,
                        "description": issue.get("Description", ""),
                        "original_text": context,
                        "fixed_text": fixed_text,
                    }
                )

        return updated_content, fixed_issues

    def _extract_snippet(self, location_text: str) -> str:
        """Extract a usable snippet from the location description."""
        # Try to find quoted text
        quotes = re.findall(r'"([^"]+)"', location_text)
        if quotes:
            return quotes[0]

        # Try to find text between single quotes
        single_quotes = re.findall(r"'([^']+)'", location_text)
        if single_quotes:
            return single_quotes[0]

        # Try to find any substantial text (at least 10 chars)
        words = re.findall(r"(\w+(?:\s+\w+){3,})", location_text)
        if words:
            for phrase in words:
                if len(phrase) >= 10:
                    return phrase

        # Fall back to any text we can find
        words = location_text.split()
        if len(words) >= 3:
            return " ".join(words[:3])

        return ""

    def _get_context(self, content: str, snippet: str) -> str:
        """Get the surrounding context of a snippet in the content."""
        # Find the snippet in the content
        snippet_idx = content.find(snippet)
        if snippet_idx == -1:
            return snippet

        # Find paragraph boundaries
        para_start = content.rfind("\n\n", 0, snippet_idx)
        if para_start == -1:
            para_start = max(0, snippet_idx - 200)

        para_end = content.find("\n\n", snippet_idx)
        if para_end == -1:
            para_end = min(len(content), snippet_idx + 200)

        # Extract the paragraph
        return content[para_start:para_end].strip()
