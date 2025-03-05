from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json

from .states import ResearchReport, ResearchIteration

class ResearchStorage:
    """Manages storage of research reports and iterations."""
    
    def __init__(self):
        # In a minimal functioning implementation, we'll use a simple in-memory storage
        self.reports = {}
        self.iterations = {}
    
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
            
        # Generate ID if not present
        report_id = report_dict.get("report_id", str(uuid.uuid4()))
        report_dict["report_id"] = report_id
        
        # Store in memory
        self.reports[report_id] = report_dict
        
        return report_id
    
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
        
        # Generate ID if not present
        iteration_id = iteration_dict.get("iteration_id", str(uuid.uuid4()))
        iteration_dict["iteration_id"] = iteration_id
            
        # Store in memory
        self.iterations[iteration_id] = iteration_dict
        
        return iteration_id
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a research report by ID.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            Research report or None if not found
        """
        return self.reports.get(report_id)
    
    async def get_project_reports(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all research reports for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of research reports
        """
        return [
            report for report in self.reports.values() 
            if report.get("project_id") == project_id
        ]
    
    async def get_report_iterations(self, report_id: str) -> List[Dict[str, Any]]:
        """Get all iterations for a research report.
        
        Args:
            report_id: ID of the report
            
        Returns:
            List of research iterations
        """
        return [
            iteration for iteration in self.iterations.values()
            if iteration.get("report_id") == report_id
        ]