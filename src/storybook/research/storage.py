from typing import Dict, Any, List, Optional
from pymongo.collection import Collection
from datetime import datetime
from .states import ResearchReport, ResearchIteration
from ..db_config import get_collection

class ResearchStorage:
    """Manages storage of research reports and iterations."""
    
    def __init__(self):
        self.reports_collection: Collection = get_collection("research_reports")
        self.iterations_collection: Collection = get_collection("research_iterations")
    
    async def store_report(self, report: ResearchReport) -> str:
        """Store a research report.
        
        Args:
            report: Research report to store
            
        Returns:
            ID of the stored report
        """
        # Convert to dictionary if needed
        if hasattr(report, "to_dict"):
            report_dict = report.to_dict()
        elif hasattr(report, "dict"):
            report_dict = report.dict()
        else:
            report_dict = dict(report)
        
        # Add timestamp if not present
        if "created_at" not in report_dict:
            report_dict["created_at"] = datetime.utcnow().isoformat()
            
        # Insert into collection
        result = await self.reports_collection.insert_one(report_dict)
        return str(result.inserted_id)
    
    async def store_iteration(self, iteration: ResearchIteration) -> str:
        """Store a research iteration.
        
        Args:
            iteration: Research iteration to store
            
        Returns:
            ID of the stored iteration
        """
        # Convert to dictionary if needed
        if hasattr(iteration, "to_dict"):
            iteration_dict = iteration.to_dict()
        elif hasattr(iteration, "dict"):
            iteration_dict = iteration.dict()
        else:
            iteration_dict = dict(iteration)
        
        # Add timestamp if not present
        if "created_at" not in iteration_dict:
            iteration_dict["created_at"] = datetime.utcnow().isoformat()
            
        # Insert into collection
        result = await self.iterations_collection.insert_one(iteration_dict)
        return str(result.inserted_id)
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a research report by ID.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            Research report or None if not found
        """
        report = await self.reports_collection.find_one({"report_id": report_id})
        return report
    
    async def get_project_reports(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all research reports for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of research reports
        """
        cursor = self.reports_collection.find({"project_id": project_id})
        return [report async for report in cursor]
    
    async def get_report_iterations(self, report_id: str) -> List[Dict[str, Any]]:
        """Get all iterations for a research report.
        
        Args:
            report_id: ID of the report
            
        Returns:
            List of research iterations
        """
        cursor = self.iterations_collection.find({"report_id": report_id})
        return [iteration async for iteration in cursor]