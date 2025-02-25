from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService
import json

class PublishingTeamSupervisor(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.team_status = {
            "consistency_checker": "idle",
            "continuity_checker": "idle",
            "editor": "idle",
            "finalisation": "idle"
        }
        self.current_draft_version = 1
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> Dict[str, Any]:
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle publishing team coordination tasks."""
        task_type = task.get("type")
        
        if task_type == "review_chapters":
            return await self._handle_chapter_review(task)
        elif task_type == "review_manuscript":
            return await self._handle_manuscript_review(task)
        elif task_type == "finalize_manuscript":
            return await self._handle_manuscript_finalization(task)
        elif task_type == "multi_writer_review":
            return await self._handle_multi_writer_review(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _handle_chapter_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the review of new chapters."""
        chapters = task.get("chapters", [])
        story_bible = task.get("story_bible", {})
        novel_id = story_bible.get("novel_id")
        
        review_report = {
            "novel_id": novel_id,
            "chapters_reviewed": [],
            "consistency_report": None,
            "continuity_report": None,
            "status": "in_progress",
            "blocking_issues": [],
            "recommendations": []
        }

        # Update team status
        self.team_status["consistency_checker"] = "working"
        self.team_status["continuity_checker"] = "working"

        # Get previous chapters for context
        previous_chapters = await self.mongodb_service.get_previous_chapters(
            novel_id,
            chapters[0].get("chapter_number", 1)
        )

        # Run consistency check
        consistency_report = await self._run_consistency_check(
            chapters,
            story_bible
        )

        # Run continuity check
        continuity_report = await self._run_continuity_check(
            chapters,
            previous_chapters,
            story_bible
        )

        # Combine reports
        review_report["consistency_report"] = consistency_report
        review_report["continuity_report"] = continuity_report
        review_report["blocking_issues"] = self._combine_blocking_issues(
            consistency_report,
            continuity_report
        )
        review_report["recommendations"] = self._combine_recommendations(
            consistency_report,
            continuity_report
        )

        # Update status
        review_report["status"] = "complete" if not review_report["blocking_issues"] else "blocked"
        
        return review_report

    async def _handle_manuscript_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the review of a complete manuscript draft."""
        manuscript = task.get("manuscript", {})
        story_bible = task.get("story_bible", {})

        review_report = {
            "novel_id": manuscript.get("novel_id"),
            "draft_version": self.current_draft_version,
            "consistency_report": None,
            "continuity_report": None,
            "editorial_report": None,
            "status": "in_progress",
            "blocking_issues": [],
            "recommendations": []
        }

        # Update team status
        self.team_status["consistency_checker"] = "working"
        self.team_status["continuity_checker"] = "working"
        self.team_status["editor"] = "working"

        # Run full manuscript consistency check
        consistency_report = await self._run_consistency_check(
            manuscript.get("chapters", []),
            story_bible
        )

        # Run full manuscript continuity check
        continuity_report = await self._run_continuity_check(
            manuscript.get("chapters", []),
            [],  # No previous chapters for full manuscript
            story_bible
        )

        # Run editorial review
        editorial_report = await self._run_editorial_review(
            manuscript,
            consistency_report,
            continuity_report
        )

        # Combine all reports
        review_report["consistency_report"] = consistency_report
        review_report["continuity_report"] = continuity_report
        review_report["editorial_report"] = editorial_report
        review_report["blocking_issues"] = self._combine_all_blocking_issues(
            consistency_report,
            continuity_report,
            editorial_report
        )
        review_report["recommendations"] = self._combine_all_recommendations(
            consistency_report,
            continuity_report,
            editorial_report
        )

        # Update status
        review_report["status"] = "complete" if not review_report["blocking_issues"] else "blocked"
        
        return review_report

    async def _handle_manuscript_finalization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the finalization of the manuscript."""
        manuscript = task.get("manuscript", {})
        style_guide = task.get("style_guide", {})

        finalization_report = {
            "novel_id": manuscript.get("novel_id"),
            "version": self.current_draft_version,
            "status": "in_progress",
            "final_manuscript": None,
            "quality_checks": None,
            "technical_checks": None
        }

        # Update team status
        self.team_status["finalisation"] = "working"

        # Run manuscript finalization
        final_manuscript = await self._run_manuscript_finalization(
            manuscript,
            style_guide
        )

        # Store finalized manuscript
        await self.mongodb_service.store_final_manuscript(
            manuscript["novel_id"],
            final_manuscript
        )

        finalization_report["final_manuscript"] = final_manuscript
        finalization_report["quality_checks"] = final_manuscript["quality_checks"]
        finalization_report["technical_checks"] = final_manuscript["formatting_report"]
        finalization_report["status"] = "complete"

        return finalization_report

    async def _handle_multi_writer_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the multi-writer review process."""
        review_report = task.get("review_report", {})
        novel_id = review_report.get("novel_id")

        # Add the report to the bible and MongoDB
        await self.mongodb_service.store_review_report(novel_id, review_report)

        # Inform the overall supervisor that the multi-writer review is complete
        await self._inform_overall_supervisor(novel_id, "multi_writer_review_complete")

        return {"status": "multi_writer_review_complete"}

    async def _run_consistency_check(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run consistency check on chapters or manuscript."""
        # Implementation to delegate to consistency checker agent
        return {}

    async def _run_continuity_check(
        self,
        chapters: List[Dict[str, Any]],
        previous_chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run continuity check on chapters or manuscript."""
        # Implementation to delegate to continuity checker agent
        return {}

    async def _run_editorial_review(
        self,
        manuscript: Dict[str, Any],
        consistency_report: Dict[str, Any],
        continuity_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run editorial review on manuscript."""
        # Implementation to delegate to editor agent
        return {}

    async def _run_manuscript_finalization(
        self,
        manuscript: Dict[str, Any],
        style_guide: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run manuscript finalization."""
        # Implementation to delegate to finalisation agent
        return {}

    def _combine_blocking_issues(
        self,
        consistency_report: Dict[str, Any],
        continuity_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine blocking issues from consistency and continuity reports."""
        blocking_issues = []
        
        if consistency_report.get("blocking_issues"):
            blocking_issues.extend(consistency_report["blocking_issues"])
            
        if continuity_report.get("blocking_issues"):
            blocking_issues.extend(continuity_report["blocking_issues"])
            
        return blocking_issues

    def _combine_recommendations(
        self,
        consistency_report: Dict[str, Any],
        continuity_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine recommendations from consistency and continuity reports."""
        recommendations = []
        
        if consistency_report.get("recommendations"):
            recommendations.extend(consistency_report["recommendations"])
            
        if continuity_report.get("recommendations"):
            recommendations.extend(continuity_report["recommendations"])
            
        return recommendations

    def _combine_all_blocking_issues(
        self,
        consistency_report: Dict[str, Any],
        continuity_report: Dict[str, Any],
        editorial_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine blocking issues from all reports."""
        blocking_issues = self._combine_blocking_issues(
            consistency_report,
            continuity_report
        )
        
        if editorial_report.get("blocking_issues"):
            blocking_issues.extend(editorial_report["blocking_issues"])
            
        return blocking_issues

    def _combine_all_recommendations(
        self,
        consistency_report: Dict[str, Any],
        continuity_report: Dict[str, Any],
        editorial_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine recommendations from all reports."""
        recommendations = self._combine_recommendations(
            consistency_report,
            continuity_report
        )
        
        if editorial_report.get("suggested_revisions"):
            recommendations.extend(editorial_report["suggested_revisions"])
            
        return recommendations

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the publishing team's work."""
        # Check if all required reports are present
        if not self._verify_report_completeness(result):
            return False

        # Validate blocking issues identification
        if not self._validate_blocking_issues(result):
            return False

        # Check recommendation quality
        if not self._validate_recommendations(result):
            return False

        # Verify team coordination
        if not self._verify_team_coordination():
            return False

        return True

    async def cleanup(self) -> None:
        """Cleanup after task completion."""
        self.team_status = {key: "idle" for key in self.team_status}
        self.state.memory = None
