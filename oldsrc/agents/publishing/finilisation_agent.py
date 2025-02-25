from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
import json

class FinalisationAgent(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.formatting_templates = self._load_formatting_templates()
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> Dict[str, Any]:
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the manuscript for publication."""
        manuscript = task.get("manuscript", {})
        style_guide = task.get("style_guide", {})

        final_manuscript = {
            "novel_id": manuscript.get("novel_id"),
            "version": manuscript.get("version", "1.0"),
            "formatted_content": await self._format_manuscript(
                manuscript,
                style_guide
            ),
            "front_matter": self._create_front_matter(manuscript),
            "back_matter": self._create_back_matter(manuscript),
            "metadata": self._generate_metadata(manuscript),
            "formatting_report": {},
            "quality_checks": {}
        }

        # Perform formatting quality checks
        final_manuscript["formatting_report"] = await self._check_formatting(
            final_manuscript
        )
        
        # Perform final quality checks
        final_manuscript["quality_checks"] = await self._perform_quality_checks(
            final_manuscript
        )

        await self.mongodb_service.store_final_manuscript(
            final_manuscript["novel_id"],
            final_manuscript
        )

        return final_manuscript

    async def _format_manuscript(
        self,
        manuscript: Dict[str, Any],
        style_guide: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the manuscript according to publishing standards."""
        formatted_content = {
            "chapters": [],
            "table_of_contents": self._generate_table_of_contents(manuscript),
            "page_layout": self._apply_page_layout(style_guide),
            "typography": self._apply_typography(style_guide)
        }
        
        # Format each chapter
        for chapter in manuscript.get("chapters", []):
            formatted_chapter = self._format_chapter(
                chapter,
                style_guide
            )
            formatted_content["chapters"].append(formatted_chapter)
        
        return formatted_content

    def _create_front_matter(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Create the front matter for the manuscript."""
        return {
            "title_page": self._create_title_page(manuscript),
            "copyright_page": self._create_copyright_page(manuscript),
            "dedication": self._create_dedication_page(manuscript),
            "table_of_contents": self._create_toc(manuscript),
            "preface": self._create_preface(manuscript)
        }

    def _create_back_matter(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Create the back matter for the manuscript."""
        return {
            "epilogue": self._create_epilogue(manuscript),
            "acknowledgments": self._create_acknowledgments(manuscript),
            "about_author": self._create_about_author(manuscript),
            "glossary": self._create_glossary(manuscript),
            "index": self._create_index(manuscript)
        }

    def _generate_metadata(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for the manuscript."""
        return {
            "title": manuscript.get("title", ""),
            "author": manuscript.get("author", ""),
            "isbn": self._generate_isbn(),
            "publication_date": "2025-02-24",
            "publisher": manuscript.get("publisher", ""),
            "genre": manuscript.get("genre", ""),
            "keywords": self._generate_keywords(manuscript),
            "description": self._generate_description(manuscript)
        }

    async def _check_formatting(
        self,
        final_manuscript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check the formatting of the final manuscript."""
        return {
            "typography_check": self._check_typography(final_manuscript),
            "layout_check": self._check_layout(final_manuscript),
            "consistency_check": self._check_formatting_consistency(final_manuscript),
            "style_guide_compliance": self._check_style_guide_compliance(final_manuscript)
        }

    async def _perform_quality_checks(
        self,
        final_manuscript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform final quality checks on the manuscript."""
        return {
            "structure_check": self._check_structure(final_manuscript),
            "content_check": self._check_content(final_manuscript),
            "metadata_check": self._check_metadata(final_manuscript),
            "technical_check": self._check_technical_requirements(final_manuscript)
        }

    def _load_formatting_templates(self) -> Dict[str, Any]:
        """Load formatting templates for different publication formats."""
        return {
            "print": self._load_print_template(),
            "ebook": self._load_ebook_template(),
            "pdf": self._load_pdf_template()
        }

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the finalization quality."""
        # Verify formatting completeness
        if not self._verify_formatting_completeness(result):
            return False

        # Check metadata completeness
        if not self._verify_metadata_completeness(result):
            return False

        # Validate technical requirements
        if not self._validate_technical_requirements(result):
            return False

        return True

    async def cleanup(self) -> None:
        """Cleanup after manuscript finalization."""
        self.state.memory = None

    def _verify_formatting_completeness(self, result: Dict[str, Any]) -> bool:
        """Verify formatting completeness."""
        required_sections = ["chapters", "table_of_contents", "page_layout", "typography"]
        formatted_content = result.get("formatted_content", {})
        return all(section in formatted_content for section in required_sections)

    def _verify_metadata_completeness(self, result: Dict[str, Any]) -> bool:
        """Verify metadata completeness."""
        required_metadata = ["title", "author", "isbn", "publication_date", "publisher", "genre", "keywords", "description"]
        metadata = result.get("metadata", {})
        return all(field in metadata for field in required_metadata)

    def _validate_technical_requirements(self, result: Dict[str, Any]) -> bool:
        """Validate technical requirements."""
        technical_checks = result.get("quality_checks", {}).get("technical_check", {})
        return technical_checks.get("status") == "pass"
