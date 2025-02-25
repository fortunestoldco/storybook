from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from services.mongodb_service import MongoDBService

class ConsistencyCheckerAgent(BaseAgent):
    """Agent responsible for checking story consistency across the manuscript."""
    
    def __init__(self, tools_service: ToolsService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.state = {
            "current_check": None,
            "issues_found": [],
            "last_check_timestamp": None
        }

    async def check_consistency(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run consistency checks on the manuscript."""
        try:
            self.logger.info("Starting consistency check")
            self.state["current_check"] = datetime.utcnow().isoformat()
            
            consistency_report = {
                "timestamp": self.state["current_check"],
                "issues": [],
                "warnings": [],
                "stats": {},
                "elements_checked": {
                    "characters": [],
                    "locations": [],
                    "plot_elements": [],
                    "world_rules": []
                }
            }

            # Check character consistency
            character_issues = await self._check_character_consistency(chapters, story_bible)
            consistency_report["issues"].extend(character_issues)
            
            # Check location consistency
            location_issues = await self._check_location_consistency(chapters, story_bible)
            consistency_report["issues"].extend(location_issues)
            
            # Check plot element consistency
            plot_issues = await self._check_plot_consistency(chapters, story_bible)
            consistency_report["issues"].extend(plot_issues)
            
            # Check world-building rules consistency
            world_issues = await self._check_world_rules_consistency(chapters, story_bible)
            consistency_report["issues"].extend(world_issues)
            
            # Calculate statistics
            consistency_report["stats"] = self._calculate_consistency_stats(
                consistency_report["issues"]
            )
            
            # Generate warnings for potential issues
            consistency_report["warnings"] = self._generate_warnings(
                consistency_report["issues"],
                consistency_report["stats"]
            )

            self.state["issues_found"] = consistency_report["issues"]
            self.state["last_check_timestamp"] = self.state["current_check"]
            
            return consistency_report

        except Exception as e:
            self.logger.error(f"Error in consistency check: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _check_character_consistency(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check character consistency throughout the manuscript."""
        issues = []
        characters = story_bible.get("characters", {})
        
        for character_name, character_data in characters.items():
            # Track character appearances and traits
            appearances = []
            trait_usage = {}
            
            for chapter_num, chapter in enumerate(chapters, 1):
                content = chapter.get("content", "")
                
                # Check character name consistency
                if character_name in content:
                    appearances.append(chapter_num)
                    
                    # Check trait consistency
                    for trait, value in character_data.get("traits", {}).items():
                        if isinstance(value, str) and value in content:
                            trait_usage[trait] = trait_usage.get(trait, []) + [chapter_num]
                
                # Check for character inconsistencies
                if self._detect_character_inconsistency(content, character_data):
                    issues.append({
                        "type": "character_inconsistency",
                        "character": character_name,
                        "chapter": chapter_num,
                        "description": f"Possible inconsistency in {character_name}'s traits or behavior",
                        "context": self._extract_context(content, character_name)
                    })
        
        return issues

    async def _check_location_consistency(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check location consistency throughout the manuscript."""
        issues = []
        locations = story_bible.get("locations", {})
        
        for location_name, location_data in locations.items():
            location_appearances = []
            
            for chapter_num, chapter in enumerate(chapters, 1):
                content = chapter.get("content", "")
                
                if location_name in content:
                    location_appearances.append(chapter_num)
                    
                    # Check location description consistency
                    if self._detect_location_inconsistency(content, location_data):
                        issues.append({
                            "type": "location_inconsistency",
                            "location": location_name,
                            "chapter": chapter_num,
                            "description": f"Possible inconsistency in {location_name}'s description or properties",
                            "context": self._extract_context(content, location_name)
                        })
        
        return issues

    async def _check_plot_consistency(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check plot element consistency throughout the manuscript."""
        issues = []
        plot_elements = story_bible.get("plot", {})
        
        # Check major plot points
        for plot_point, plot_data in plot_elements.get("major_points", {}).items():
            expected_chapter = plot_data.get("chapter")
            if expected_chapter:
                content = chapters[expected_chapter - 1].get("content", "")
                if not self._verify_plot_point(content, plot_data):
                    issues.append({
                        "type": "plot_inconsistency",
                        "element": plot_point,
                        "chapter": expected_chapter,
                        "description": "Major plot point missing or inconsistent",
                        "expected": plot_data.get("description")
                    })
        
        # Check plot arcs
        for arc_name, arc_data in plot_elements.get("arcs", {}).items():
            arc_issues = self._check_plot_arc_consistency(chapters, arc_data)
            issues.extend(arc_issues)
        
        return issues

    async def _check_world_rules_consistency(
        self,
        chapters: List[Dict[str, Any]],
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency with established world rules."""
        issues = []
        world_rules = story_bible.get("world_building", {}).get("rules", {})
        
        for rule_name, rule_data in world_rules.items():
            for chapter_num, chapter in enumerate(chapters, 1):
                content = chapter.get("content", "")
                
                if self._detect_world_rule_violation(content, rule_data):
                    issues.append({
                        "type": "world_rule_violation",
                        "rule": rule_name,
                        "chapter": chapter_num,
                        "description": f"Possible violation of world rule: {rule_name}",
                        "context": self._extract_context(content, rule_name)
                    })
        
        return issues

    def _detect_character_inconsistency(
        self,
        content: str,
        character_data: Dict[str, Any]
    ) -> bool:
        """Detect inconsistencies in character portrayal."""
        # Example logic: Check for inconsistencies in character traits
        for trait, value in character_data.get("traits", {}).items():
            if isinstance(value, str) and value not in content:
                return True
        return False

    def _detect_location_inconsistency(
        self,
        content: str,
        location_data: Dict[str, Any]
    ) -> bool:
        """Detect inconsistencies in location descriptions."""
        # Example logic: Check for inconsistencies in location properties
        for property, value in location_data.get("properties", {}).items():
            if isinstance(value, str) and value not in content:
                return True
        return False

    def _verify_plot_point(
        self,
        content: str,
        plot_data: Dict[str, Any]
    ) -> bool:
        """Verify if a plot point is properly executed."""
        # Example logic: Check for presence of key plot elements
        for element in plot_data.get("elements", []):
            if element not in content:
                return False
        return True

    def _check_plot_arc_consistency(
        self,
        chapters: List[Dict[str, Any]],
        arc_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of a plot arc across chapters."""
        issues = []
        arc_points = arc_data.get("points", [])
        
        for i in range(1, len(arc_points)):
            current_point = arc_points[i]
            previous_point = arc_points[i - 1]
            
            current_chapter = chapters[current_point["chapter"] - 1].get("content", "")
            previous_chapter = chapters[previous_point["chapter"] - 1].get("content", "")
            
            if not self._verify_plot_point(current_chapter, current_point):
                issues.append({
                    "type": "plot_arc_inconsistency",
                    "arc": arc_data.get("name"),
                    "chapter": current_point["chapter"],
                    "description": f"Inconsistency in plot arc: {arc_data.get('name')}",
                    "context": self._extract_context(current_chapter, current_point["description"])
                })
        
        return issues

    def _detect_world_rule_violation(
        self,
        content: str,
        rule_data: Dict[str, Any]
    ) -> bool:
        """Detect violations of established world rules."""
        # Example logic: Check for violations of world rules
        for rule, value in rule_data.items():
            if isinstance(value, str) and value not in content:
                return True
        return False

    def _extract_context(self, content: str, keyword: str, window: int = 100) -> str:
        """Extract surrounding context for an issue."""
        try:
            start_idx = content.find(keyword)
            if start_idx == -1:
                return ""
            
            start = max(0, start_idx - window)
            end = min(len(content), start_idx + len(keyword) + window)
            
            return content[start:end].strip()
        except Exception:
            return ""

    def _calculate_consistency_stats(
        self,
        issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics about consistency issues."""
        stats = {
            "total_issues": len(issues),
            "issues_by_type": {},
            "issues_by_chapter": {},
            "severity_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        for issue in issues:
            # Count by type
            issue_type = issue.get("type", "unknown")
            stats["issues_by_type"][issue_type] = stats["issues_by_type"].get(issue_type, 0) + 1
            
            # Count by chapter
            chapter = issue.get("chapter", 0)
            stats["issues_by_chapter"][chapter] = stats["issues_by_chapter"].get(chapter, 0) + 1
            
            # Count by severity (could be determined by issue type or other factors)
            severity = self._determine_issue_severity(issue)
            stats["severity_distribution"][severity] += 1
        
        return stats

    def _determine_issue_severity(self, issue: Dict[str, Any]) -> str:
        """Determine the severity of an issue."""
        # Implementation would classify issue severity
        return "medium"  # Placeholder

    def _generate_warnings(
        self,
        issues: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate warnings based on issues and statistics."""
        warnings = []
        
        # Check for high concentration of issues
        for chapter, count in stats["issues_by_chapter"].items():
            if count > 5:  # Arbitrary threshold
                warnings.append({
                    "type": "high_issue_concentration",
                    "chapter": chapter,
                    "description": f"High concentration of consistency issues in chapter {chapter}"
                })
        
        # Check for systematic issues
        for issue_type, count in stats["issues_by_type"].items():
            if count > len(issues) * 0.3:  # If issue type represents >30% of all issues
                warnings.append({
                    "type": "systematic_issue",
                    "issue_type": issue_type,
                    "description": f"Systematic {issue_type} issues detected across manuscript"
                })
        
        return warnings

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the quality of the consistency check."""
        try:
            if "error" in result:
                return False
                
            required_fields = ["issues", "warnings", "stats", "elements_checked"]
            if not all(field in result for field in required_fields):
                return False
                
            if not isinstance(result["issues"], list):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in reflection: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Cleanup after consistency check."""
        try:
            self.state = {
                "current_check": None,
                "issues_found": [],
                "last_check_timestamp": None
            }
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
