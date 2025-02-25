from agents.research.character_research_agent import CharacterResearchAgent
from agents.research.market_research_agent import MarketResearchAgent
from agents.research.consumer_insights_agent import ConsumerInsightsAgent
from agents.writing.content_generation_agent import ContentGenerationAgent
from agents.writing.character_development_agent import CharacterDevelopmentAgent
from agents.writing.world_building_agent import WorldBuildingAgent
from agents.publishing.editor_agent import EditorAgent
from agents.publishing.consistency_checker_agent import ConsistencyCheckerAgent
from agents.publishing.conflict_resolution_agent import ConflictResolutionAgent
from agents.publishing.thematic_analysis_agent import ThematicAnalysisAgent
from agents.publishing.narrative_structure_agent import NarrativeStructureAgent
from supervisors.team_supervisor import TeamSupervisor
from supervisors.overall_supervisor import OverallSupervisor

class StorybookGraph:
    def __init__(self, tools_service):
        self.character_research_agent = CharacterResearchAgent(tools_service)
        self.market_research_agent = MarketResearchAgent(tools_service)
        self.consumer_insights_agent = ConsumerInsightsAgent(tools_service)
        self.content_generation_agent = ContentGenerationAgent(tools_service)
        self.character_development_agent = CharacterDevelopmentAgent(tools_service)
        self.world_building_agent = WorldBuildingAgent(tools_service)
        self.editor_agent = EditorAgent(tools_service)
        self.consistency_checker_agent = ConsistencyCheckerAgent(tools_service)
        self.conflict_resolution_agent = ConflictResolutionAgent(tools_service)
        self.thematic_analysis_agent = ThematicAnalysisAgent(tools_service)
        self.narrative_structure_agent = NarrativeStructureAgent(tools_service)
        self.team_supervisor = TeamSupervisor(tools_service)
        self.overall_supervisor = OverallSupervisor(tools_service)

    async def run(self, project_details):
        await self.overall_supervisor.handle_task({
            "type": "initiate_project",
            "project_details": project_details
        })
        market_data = await self.market_research_agent.handle_task({"market_data": "Collected market data"})
        consumer_insights = await self.consumer_insights_agent.handle_task({"survey_responses": "Collected survey responses"})
        themes = market_data["analysis"]["trending_genres"]
        character_profiles = await self.character_research_agent.handle_task({"type": "profile", "character_info": "Character details"})
        chapter_outline = await self.content_generation_agent.handle_task({
            "themes": themes,
            "character_profiles": character_profiles["profile"]
        })
        character_arc = await self.character_development_agent.handle_task({
            "character_profile": "Profile of main character"
        })
        world_details = await self.world_building_agent.handle_task({
            "world_description": "Description of the world setting and themes"
        })
        manuscript = f"{chapter_outline['chapter_outline']}\n{character_arc['character_arc']}\n{world_details['world_details']}"
        edited_text = await self.editor_agent.handle_task({"manuscript": manuscript})
        consistency_report = await self.consistency_checker_agent.handle_task({"manuscript": edited_text["edited_text"]})
        conflict_analysis = await self.conflict_resolution_agent.handle_task({
            "type": "analyze_conflict",
            "interactions": "Character interactions"
        })
        resolution_strategy = await self.conflict_resolution_agent.handle_task({
            "type": "resolve_conflict",
            "conflict_details": conflict_analysis["conflict_analysis"]
        })
        thematic_analysis = await self.thematic_analysis_agent.handle_task({
            "type": "analyze_theme",
            "manuscript": edited_text["edited_text"]
        })
        thematic_improvement = await self.thematic_analysis_agent.handle_task({
            "type": "improve_theme",
            "identified_themes": thematic_analysis["thematic_analysis"]
        })
        structure_analysis = await self.narrative_structure_agent.handle_task({
            "type": "analyze_structure",
            "manuscript": edited_text["edited_text"]
        })
        structure_improvement = await self.narrative_structure_agent.handle_task({
            "type": "improve_structure",
            "structure_analysis": structure_analysis["structure_analysis"]
        })
        final_manuscript = f"{edited_text['edited_text']}\n{thematic_improvement['theme_improvement']}\n{structure_improvement['structure_improvement']}"
        return final_manuscript
