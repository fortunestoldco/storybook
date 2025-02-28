from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
import loggingport Dict, List, Any, Optional
import jsoning
import reon
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import DocumentTemplate
from langchain_openai import OpenAIEmbeddingsOutputParser
from langchain_core.runnables import RunnablePassthrough
from storybook.agents.base import BaseAgentnt
from storybook.config import create_llm, get_llm
from storybook.db.document_store import DocumentStore
from storybook.db.mongodb_store import MongoDBStore  # Add missing import
from storybook.tools.document_tools import DocumentTools
from storybook.db.document_store import DocumentStore
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

class ContentAnalyzer(BaseAgent):
    """Agent responsible for analyzing manuscript content."""
    """Agent responsible for maintaining narrative continuity."""
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional LLM configuration."""
        super().__init__(llm_config)
        self.document_store = DocumentStore()
        self.document_tools = DocumentTools())
        self.embeddings = OpenAIEmbeddings()
        self.db = MongoDBStore()    def check_continuity(

    def get_tools(self):
        """Get tools available to this agent."""udience: Optional[Dict[str, Any]] = None,
        return [None,
            self.document_tools.get_manuscript_tool(),
            self.document_tools.get_manuscript_search_tool(),Dict[str, Any]:
        ]        """Check and maintain narrative continuity."""

    def analyze_content(self, manuscript_id: str, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze manuscript content with optional runtime LLM configuration."""if llm_config:
        try:g)
            # Update LLM if new config provided
            if llm_config:manuscript(manuscript_id)
                self.llm = create_llm(llm_config)ot manuscript:
                
            logger.info(f"Analyzing content for manuscript {manuscript_id}")
            
            # Get manuscript content from document store
            document = self.document_store.get_manuscript(manuscript_id)content"],
            if not document:
                raise ValueError(f"Manuscript {manuscript_id} not found"))
            
            content = document.get("content", "")# If issues found, attempt to fix them
            
            # Perform comprehensive content analysisntent = self._fix_continuity_issues(
            analysis = {
                "sentiment": self._analyze_sentiment(content),
                "readability": self._analyze_readability(content),
                "content_structure": self._analyze_structure(content),
                "genre_match": self._analyze_genre(content),
                "themes": self._identify_themes(content),pt with fixed content
                "key_elements": {
                    "plot_points": self._extract_plot_points(content),
                    "characters": self._extract_characters(content),
                    "settings": self._extract_settings(content),
                }
            }    return {
            "manuscript_id": manuscript_id,
            return {en(issues["issues"]),
                "analysis": analysis,
                "message": "Content analysis completed successfully"       "continuity_analysis": issues
            }
        except Exception as e:
            logger.error(f"Error analyzing content for manuscript {manuscript_id}: {e}")
            return {"error": str(e)}                "manuscript_id": manuscript_id,

    def analyze_progress(self, manuscript_id: str, previous_analysis: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Analyze progress made after a specific stage of the transformation."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            logger.error(f"Manuscript {manuscript_id} not found")ion as e:
            return {}            logger.error(f"Error in check_continuity: {str(e)}")

        content = manuscript.get("content", "")
str) -> Dict[str, Any]:
        # Define what to analyze based on the stage issues in the manuscript."""
        progress_analysis = {}        # Break the content into manageable chunks for analysis
len(content) // 3)
        if stage == "story_arc":
            # Analyze structure improvements
            new_structure = self._analyze_structure(content)
            progress_analysis["structure_improvements"] = self._compare_analysis(
                previous_analysis.get("content_structure", {}), new_structurehunks.append(chunk)
            )
            progress_analysis["content_structure"] = new_structure        # Limit to a reasonable number of chunks

        elif stage == "language":
            # Analyze style and readability improvements
            new_style = self._analyze_style(content)
            new_readability = self._analyze_readability(content)            """
es in manuscripts.
            progress_analysis["style_improvements"] = self._compare_analysis(
                previous_analysis.get("writing_style", {}), new_styleze the following manuscript excerpts for continuity issues such as:
            )
            progress_analysis["readability_improvements"] = self._compare_analysis(g)
                previous_analysis.get("readability", {}), new_readabilitymeline inconsistencies (events out of order, impossible timing)
            )        3. Setting inconsistencies (locations changing unexpectedly)
nformation)
            progress_analysis["writing_style"] = new_styletuations)
            progress_analysis["readability"] = new_readability        

        elif stage == "complete":
            # Comprehensive final analysis
            new_sentiment = self._analyze_sentiment(content)
            new_readability = self._analyze_readability(content)
            new_structure = self._analyze_structure(content)nconsistency
            new_style = self._analyze_style(content)        3. Location: Where in the excerpt the issue appears (quote the relevant text)
ts the story (High, Medium, Low)
            # Compare all metrics with originallve the inconsistency
            progress_analysis["overall_improvements"] = {
                "sentiment": self._compare_analysis(s.
                    previous_analysis.get("sentiment", {}), new_sentiment on clear continuity issues, not subjective writing preferences.
                ),
                "readability": self._compare_analysis(
                    previous_analysis.get("readability", {}), new_readability
                ),
                "structure": self._compare_analysis(
                    previous_analysis.get("content_structure", {}), new_structurenk in enumerate(chunks):
                ),
                "style": self._compare_analysis( way through):\n{chunk}"
                    previous_analysis.get("writing_style", {}), new_style
                ),
            }        combined_excerpts = "\n\n" + "\n\n".join(excerpts_text)

            # Generate improvement summary
            progress_analysis["improvement_summary"] = (
                self._generate_improvement_summary(mbda _: combined_excerpts}
                    previous_analysis,
                    {
                        "sentiment": new_sentiment,
                        "readability": new_readability,
                        "content_structure": new_structure,
                        "writing_style": new_style,n
                    },r = chain.invoke("Identify continuity issues")
                )
            )        # Parse the issues

        # Store the progress analysis
        analysis_doc_id = self.document_store.store_analysis_document(
            manuscript_id, f"{stage}_progress_analysis", progress_analysis       return {"issues": issues_list}
        )            return {"issues": []}

        # Store in MongoDB Atlas Vector for easy searchingr(f"Failed to parse continuity issues JSON: {issues_str}")
        doc = Document(
            page_content=json.dumps(progress_analysis, indent=2),xtract structured data manually
            metadata={
                "type": "progress_analysis",
                "stage": stage,
                "manuscript_id": manuscript_id,
                "analysis_id": analysis_doc_id,  r"(\d+\.\s+Issue Type:.*?)(?=\d+\.\s+Issue Type:|$)",
            },       issues_str,
        )                re.DOTALL,

        self.document_store.db.store_documents_with_embeddings("analysis", [doc])
locks:
        return progress_analysis                issue = {}

    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze emotional tone and sentiment of the content."""arch(r"Issue Type:?\s*(.*?)(?=\n|$)", block)
        # Implementation using LLMf type_match:
        return {}                    issue["Issue Type"] = type_match.group(1).strip()

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze readability metrics."""arch(
        # Implementation using LLM   r"Description:?\s*(.*?)(?=\n\d|$)", block, re.DOTALL
        return {}                )

    def _analyze_structure(self, content: str) -> Dict[str, Any]: desc_match.group(1).strip()
        """Analyze narrative structure."""
        # Implementation using LLM Extract location
        return {}                loc_match = re.search(r"Location:?\s*(.*?)(?=\n\d|$)", block, re.DOTALL)

    def _analyze_genre(self, content: str) -> Dict[str, Any]:tch.group(1).strip()
        """Determine genre characteristics."""
        # Implementation using LLM Extract severity
        return {}                sev_match = re.search(r"Severity:?\s*(.*?)(?=\n|$)", block)

    def _identify_themes(self, content: str) -> List[str]:oup(1).strip()
        """Extract major themes from the content."""
        # Implementation using LLM Extract recommended fix
        return []                fix_match = re.search(
ALL
    def _extract_plot_points(self, content: str) -> List[Dict[str, Any]]:
        """Extract major plot points."""
        # Implementation using LLM   issue["Recommended Fix"] = fix_match.group(1).strip()
        return []

    def _extract_characters(self, content: str) -> List[Dict[str, Any]]:
        """Extract character mentions and descriptions."""
        # Implementation using LLMn {"issues": issues}
        return []

    def _extract_settings(self, content: str) -> List[Dict[str, Any]]:es: Dict[str, Any]
        """Extract setting descriptions."""
        # Implementation using LLMentified continuity issues in the manuscript."""
        return []        issues = continuity_issues.get("issues", [])

    def _compare_analysis(
        self, previous: Dict[str, Any], current: Dict[str, Any]= [
    ) -> Dict[str, Any]:
        """Compare previous and current analysis to determine improvements.""" issues
        improvements = {, "").lower() in ["high", "medium"]
            "improved_aspects": [],
            "new_aspects": [],
            "changed_aspects": [],rn content unchanged
            "improvement_score": 0.0,f not priority_issues:
        }            return content, []

        # Skip if either analysis is empty
        if not previous or not current:nt
            return improvements        fixed_issues = []

        # Track improvement count for scoring_issues:
        improvement_count = 0 issue information
        total_aspects = 0            issue_type = issue.get("Issue Type", "")
 = issue.get("Location", "")
        # Compare each key)
        all_keys = set(previous.keys()) | set(current.keys())
        for key in all_keys: a location or fix, skip
            total_aspects += 1            if not issue_location or not recommended_fix:

            # New aspect
            if key not in previous and key in current:ontent
                improvements["new_aspects"].append(key) from the location
                improvement_count += 1            location_snippet = self._extract_snippet(issue_location)

            # Changed aspectpet not in updated_content:
            elif key in previous and key in current:
                prev_value = previous[key]
                curr_value = current[key]            # Create prompt for fixing this specific issue
rom_template(
                # If values are different
                if prev_value != curr_value:nconsistency in a manuscript.
                    improvements["changed_aspects"].append(
                        {"aspect": key, "previous": prev_value, "current": curr_value}e: {issue_type}
                    )            Issue Description: {issue_description}

                    # Determine if it's an improvement
                    if self._is_improvement(key, prev_value, curr_value):
                        improvements["improved_aspects"].append(key)
                        improvement_count += 1            Recommended Fix Approach:

        # Calculate overall improvement score (0.0-1.0)
        if total_aspects > 0:inuity issue.
            improvements["improvement_score"] = improvement_count / total_aspects            Make minimal changes - only what's necessary to resolve the inconsistency.
me style, tone, and length as the original text.
        return improvements            
matic section.
    def _is_improvement(self, aspect: str, previous: Any, current: Any) -> bool:
        """Determine if a change represents an improvement."""
        # For numeric values
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            # For readability scores, higher is generally better
            if "score" in aspect or "count" in aspect:
                return current > previous                    "issue_type": lambda _: issue_type,
", ""),
        # For text values, longer descriptions might indicate more detailt(
        if isinstance(previous, str) and isinstance(current, str):
            # If complexity or readability, prefer more moderate values
            if "complexity" in aspect:
                return current.lower() in ["moderate", "balanced"]                }

            # For most text aspects, more detail is better
            return len(current) > len(previous) * 1.1  # 10% longer                | StrOutputParser()

        # For lists, more items might indicate more comprehensive analysis
        if isinstance(previous, list) and isinstance(current, list):
            return len(current) > len(previous)            fixed_text = chain.invoke("Fix continuity issue")

        # Default - assume change is an improvementce the problematic text with the fixed version
        return True            context = self._get_context(updated_content, location_snippet)
ent:
    def _generate_improvement_summary(ntext, fixed_text, 1)
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of improvements from initial to final analysis."""
        # Define prompt for improvement summary
        prompt = ChatPromptTemplate.from_template(         "issue_type": issue_type,
            """
        You are a Manuscript Improvement Analyst. Compare the initial and final analysis of a manuscript
        and summarize the improvements made during the transformation process.                "fixed_text": fixed_text,
        
        Initial Analysis:
        {initial_analysis}
        content, fixed_issues
        Final Analysis:
        {final_analysis}_extract_snippet(self, location_text: str) -> str:
        om the location description."""
        Provide a detailed summary of:
        1. Overall improvement assessment, location_text)
        2. Most significant improvements
        3. Areas with the most dramatic changes
        4. Remaining opportunities for improvement
        5. Impact on target audience appeal# Try to find text between single quotes
        on_text)
        Format your response as a comprehensive report.single_quotes:
        """   return single_quotes[0]
        )
ial text (at least 10 chars)
        # Format the analysis datalocation_text)
        initial_str = json.dumps(initial, indent=2)
        final_str = json.dumps(final, indent=2)            for phrase in words:
ase) >= 10:
        # Create the chain   return phrase
        chain = (
            {
                "initial_analysis": lambda _: initial_str,
                "final_analysis": lambda _: final_str,n(words) >= 3:
            } ".join(words[:3])
            | prompt
            | self.llm
            | StrOutputParser()
        )    def _get_context(self, content: str, snippet: str) -> str:
ounding context of a snippet in the content."""
        # Run the chain
        summary = chain.invoke("Generate improvement summary")        snippet_idx = content.find(snippet)
et_idx == -1:
        return {
            "improvement_summary": summary,
            "improvement_metrics": {
                "sentiment_change": self._calculate_metric_change(
                    initial.get("sentiment", {}), final.get("sentiment", {})art == -1:
                ),
                "readability_change": self._calculate_metric_change(
                    initial.get("readability", {}), final.get("readability", {}) content.find("\n\n", snippet_idx)
                ),
                "structure_change": self._calculate_metric_change(0)
                    initial.get("content_structure", {}),
                    final.get("content_structure", {}),the paragraph
                ),
                "style_change": self._calculate_metric_change(                    initial.get("writing_style", {}), final.get("writing_style", {})                ),            },        }    def _calculate_metric_change(        self, initial: Dict[str, Any], final: Dict[str, Any]    ) -> float:        """Calculate the percentage of change between initial and final metrics."""        if not initial or not final:            return 0.0        total_fields = 0        changed_fields = 0        for key in initial:            if key in final:                total_fields += 1                if initial[key] != final[key]:                    changed_fields += 1        return changed_fields / max(total_fields, 1)    def _get_timestamp(self) -> str:        """Get current timestamp in ISO format."""        from datetime import datetime        return datetime.now().isoformat()    def _store_analysis_document(self, manuscript_id: str, analysis: Dict[str, Any]) -> str:        """Store analysis in vector store."""        try:            doc = Document(                page_content=json.dumps(analysis, indent=2),                metadata={                    "type": "content_analysis",                    "manuscript_id": manuscript_id,                    "timestamp": self._get_timestamp()                }            )            return self.document_store.db.store_documents_with_embeddings("analysis", [doc])        except Exception as e:            logger.error(f"Error storing analysis: {e}")            return None
