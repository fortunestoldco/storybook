from typing import Dict, List, Any, Optional
import logging

from langchain_core.tools import Tool

from storybook.db.document_store import DocumentStore

logger = logging.getLogger(__name__)


class DocumentTools:
    """Tools for working with manuscript documents."""

    def __init__(self):
        self.document_store = DocumentStore()

    def get_manuscript_tool(self):
        """Create a tool to retrieve a manuscript."""
        return Tool(
            name="GetManuscript",
            description="Retrieves the full text of a manuscript by ID.",
            func=self._get_manuscript,
        )

    def get_manuscript_search_tool(self):
        """Create a tool to search for relevant parts of a manuscript."""
        return Tool(
            name="SearchManuscript",
            description="Searches for relevant sections within a manuscript using semantic search.",
            func=self._search_manuscript,
        )

    def _get_manuscript(self, manuscript_id: str) -> str:
        """Retrieve the full text of a manuscript."""
        manuscript = self.document_store.get_manuscript(manuscript_id)
        if not manuscript:
            return f"Error: Manuscript with ID {manuscript_id} not found."

        return manuscript.get("content", "")

    def _search_manuscript(
        self, query: str, manuscript_id: Optional[str] = None, k: int = 5
    ) -> str:
        """Search for relevant parts of a manuscript."""
        if not manuscript_id:
            # Extract manuscript_id from the query if not provided explicitly
            import re

            match = re.search(r"manuscript[\s_-]?id[:\s]+([a-zA-Z0-9]+)", query)
            if match:
                manuscript_id = match.group(1)
            else:
                return "Error: Please provide a manuscript_id to search within."

        try:
            # Get relevant parts of the manuscript
            results = self.document_store.get_manuscript_relevant_parts(
                manuscript_id, query, k
            )

            if not results:
                return f"No relevant sections found in manuscript {manuscript_id} for query: {query}"

            # Format the results
            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(f"Section {i}:\n{doc.page_content}\n")

            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error searching manuscript: {e}")
            return f"Error searching manuscript: {str(e)}"