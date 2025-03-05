from typing import Dict, Any, List, Optional
from pymongo.collection import Collection
from .models import ResearchReport, ResearchIteration
from ..db_config import get_collection

class ResearchStorage:
    """Manages storage of research reports and iterations."""
    
    def __init__(self):
        self.reports_collection: Collection = get_collection("research_reports")
        self.iterations_collection: Collection = get_collection("research_iterations")
    
    async def store_report(self, report: ResearchReport) -> str:
        """Store a research report."""
        result = await self.reports_collection.insert_one(report.to_dict())
        return str(result.inserted_id)
    
    async def store_iteration(self, iteration: ResearchIteration) -> str:
        """Store a research iteration."""
        result = await self.iterations_collection.insert_one(iteration.to_dict())
        return str(result.inserted_id)
    
    async def get_report(self, report_id: str) -> Optional[ResearchReport]:
        """Retrieve a research report by ID."""
        data = await self.reports_collection.find_one({"report_id": report_id})
        return ResearchReport.from_dict(data) if data else None
    
    async def get_project_reports(self, project_id: str) -> List[ResearchReport]:
        """Get all research reports for a project."""
        cursor = self.reports_collection.find({"project_id": project_id})
        return [ResearchReport.from_dict(doc) async for doc in cursor]
    
    async def get_report_iterations(self, report_id: str) -> List[ResearchIteration]:
        """Get all iterations for a research report."""
        cursor = self.iterations_collection.find({"report_id": report_id})
        return [ResearchIteration.from_dict(doc) async for doc in cursor]