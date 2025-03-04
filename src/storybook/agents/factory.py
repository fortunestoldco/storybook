from typing import Dict, Any, Callable, List
from langchain_core.runnables import RunnableConfig

from storybook.configuration import Configuration
from storybook.state import NovelSystemState
from storybook.utils import load_chat_model
from storybook.prompts import get_agent_prompt
from storybook.tools import *

class AgentFactory:
    """Factory for creating specialized novel writing agents."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.base_model = load_chat_model(config.model)
        self._init_tool_registry()
    
    def _init_tool_registry(self) -> None:
        """Initialize the tool registry for each agent."""
        self.tool_registry = {
            # Initialization Phase
            "executive_director": [QualityAssessmentTool(), TaskDelegationTool()],
            "human_feedback_manager": [FeedbackProcessingTool()],
            "quality_assessment_director": [QualityMetricsTool(), QualityGateTool()],
            
            # Development Phase
            "creative_director": [StoryElementsTool(), CreativeAssessmentTool()],
            "structure_architect": [StoryStructureTool(), PacingAnalysisTool(), ChapterOutlineTool()],
            "plot_development_specialist": [PlotThreadTool(), ConflictDesignTool(), PlotCoherenceTool()],
            "world_building_expert": [WorldDesignTool(), ConsistencyCheckerTool(), LocationManagerTool()],
            
            # Creation Phase
            "content_development_director": [ContentPlanningTool(), ContentQualityTool(), ContentProgressTool()],
            "chapter_drafter": [ChapterStructureTool(), SceneSequenceTool(), NarrativeFlowTool()],
            "dialogue_crafter": [DialogueGenerationTool(), CharacterVoiceTool(), SubtextTool()],
            
            # Refinement Phase
            "editorial_director": [EditorialPlanningTool(), QualityAssessmentTool(), RevisionCoordinationTool()],
            "prose_enhancement_specialist": [StyleRefinementTool(), ImageryEnhancementTool(), SentenceStructureTool()],
            
            # Finalization Phase
            "formatting_standards_expert": [FormatValidationTool(), StyleGuideComplianceTool(), PublishingStandardsTool()],
            "positioning_specialist": [MarketAnalysisTool(), PositioningStrategyTool(), CompetitorAnalysisTool()]
        }
    
    def create_agent(self, agent_name: str, project_id: str) -> Callable:
        """Create an agent function for the specified role."""
        if agent_name not in self.config.agent_roles:
            raise ValueError(f"Unknown agent role: {agent_name}")
        
        tools = self.tool_registry.get(agent_name, [])
        
        async def agent_function(
            state: NovelSystemState, 
            config: RunnableConfig
        ) -> Dict[str, Any]:
            configuration = Configuration.from_runnable_config(config)
            system_prompt = get_agent_prompt(agent_name, project_id, state)
            
            # Execute agent with its tools
            result = await self._execute_agent(
                agent_name=agent_name,
                tools=tools,
                state=state,
                system_prompt=system_prompt,
                config=configuration
            )
            
            return result
        
        return agent_function
    
    async def _execute_agent(
        self,
        agent_name: str,
        tools: List[NovelWritingTool],
        state: NovelSystemState,
        system_prompt: str,
        config: Configuration
    ) -> Dict[str, Any]:
        """Execute the agent with its tools."""
        # Basic implementation - to be expanded
        try:
            task = state.current_input.get("task", "")
            result = {"messages": []}
            
            for tool in tools:
                if tool.should_handle(task):
                    tool_result = await tool._arun(
                        content=state.project.content,
                        task=task,
                        config=config
                    )
                    result.update(tool_result)
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "messages": [{"role": "system", "content": f"Error in {agent_name}: {str(e)}"}]
            }