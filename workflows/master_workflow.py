# storybook/workflows/master_workflow.py

"""
Master workflow for the Storybook novel generation system.
"""

from typing import Dict, Any, List, Optional, Callable, Union, TypedDict
from enum import Enum
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from storybook.states.project_state import ProjectState, ProjectStatus
from storybook.agents.base_agent import BaseAgent

class MasterWorkflowState(TypedDict):
    """Type definition for master workflow state."""
    project_state: Dict[str, Any]
    current_phase: str
    phase_output: Optional[Dict[str, Any]]
    agent_outputs: Dict[str, Any]
    errors: List[str]

class WorkflowPhase(str, Enum):
    """Enum for workflow phases."""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    PLANNING = "planning"
    CHARACTER_DEVELOPMENT = "character_development"
    DRAFTING = "drafting"
    REVISION = "revision"
    OPTIMIZATION = "optimization"
    PUBLICATION = "publication"

class MasterWorkflow:
    """Master workflow for novel generation process."""
    
    def __init__(
        self, 
        agents: Dict[str, BaseAgent], 
        config: Dict[str, Any] = None
    ):
        """
        Initialize the master workflow.
        
        Args:
            agents: Dictionary of agents to use in the workflow
            config: Configuration for the workflow
        """
        self.agents = agents
        self.config = config or {}
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the workflow graph.
        
        Returns:
            StateGraph instance
        """
        # Create the workflow graph
        workflow = StateGraph(MasterWorkflowState)
        
        # Add nodes for each major phase
        workflow.add_node(WorkflowPhase.INITIALIZATION, self._run_initialization)
        workflow.add_node(WorkflowPhase.RESEARCH, self._run_research)
        workflow.add_node(WorkflowPhase.PLANNING, self._run_planning)
        workflow.add_node(WorkflowPhase.CHARACTER_DEVELOPMENT, self._run_character_development)
        workflow.add_node(WorkflowPhase.DRAFTING, self._run_drafting)
        workflow.add_node(WorkflowPhase.REVISION, self._run_revision)
        workflow.add_node(WorkflowPhase.OPTIMIZATION, self._run_optimization)
        workflow.add_node(WorkflowPhase.PUBLICATION, self._run_publication)
        
        # Define conditional edges
        workflow.add_conditional_edges(
            WorkflowPhase.INITIALIZATION,
            self._determine_next_phase,
            {
                WorkflowPhase.RESEARCH: WorkflowPhase.RESEARCH,
                WorkflowPhase.PLANNING: WorkflowPhase.PLANNING,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.RESEARCH,
            self._determine_next_phase,
            {
                WorkflowPhase.PLANNING: WorkflowPhase.PLANNING,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.PLANNING,
            self._determine_next_phase,
            {
                WorkflowPhase.CHARACTER_DEVELOPMENT: WorkflowPhase.CHARACTER_DEVELOPMENT,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.CHARACTER_DEVELOPMENT,
            self._determine_next_phase,
            {
                WorkflowPhase.DRAFTING: WorkflowPhase.DRAFTING,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.DRAFTING,
            self._determine_next_phase,
            {
                WorkflowPhase.REVISION: WorkflowPhase.REVISION,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.REVISION,
            self._determine_next_phase,
            {
                WorkflowPhase.REVISION: WorkflowPhase.REVISION,
                WorkflowPhase.OPTIMIZATION: WorkflowPhase.OPTIMIZATION,
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            WorkflowPhase.OPTIMIZATION,
            self._determine_next_phase,
            {
                WorkflowPhase.PUBLICATION: WorkflowPhase.PUBLICATION,
                END: END
            }
        )
        
        workflow.add_edge(WorkflowPhase.PUBLICATION, END)
        
        # Set the entry point
        workflow.set_entry_point(WorkflowPhase.INITIALIZATION)
        
        return workflow
    
    def _run_initialization(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run initialization phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Create a project state if not exists
        project_state = state.get("project_state", {})
        if not project_state:
            # Initialize a new project state
            project_state_manager = ProjectState()
            project_state = project_state_manager.get_state()
        else:
            # Use existing project state
            project_state_manager = ProjectState(project_state)
            
        # Update project status
        project_state_manager.update_status(ProjectStatus.INITIALIZED)
            
        # Run the project lead agent
        if "project_lead" in self.agents:
            project_lead_output = self.agents["project_lead"].run({
                "task": "initialize_project",
                "project_parameters": state.get("project_parameters", {})
            })
            
            # Update project with initialization results
            if "project_updates" in project_lead_output["output"]:
                project_state_manager.update_state(project_lead_output["output"]["project_updates"])
                
            # Store agent output
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["project_lead"] = project_lead_output
        else:
            agent_outputs = state.get("agent_outputs", {})
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.INITIALIZATION,
            "phase_output": {
                "initialized": True,
                "project_id": project_state_manager.get_state()["project_id"]
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_research(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run research phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.RESEARCHING)
        
        # Run the market research agent
        agent_outputs = state.get("agent_outputs", {})
        if "market_research" in self.agents:
            market_research_output = self.agents["market_research"].run({
                "task": "research_market",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with research results
            if "market_research" in market_research_output["output"]:
                project_state_manager.update_state({
                    "market_research": market_research_output["output"]["market_research"]
                })
                
            # Store agent output
            agent_outputs["market_research"] = market_research_output
            
        # Run other research agents as needed
        # (Additional research agents would be added here)
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.RESEARCH,
            "phase_output": {
                "research_completed": True
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_planning(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run planning phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.PLANNING)
        
        # Run the structure specialist agent
        agent_outputs = state.get("agent_outputs", {})
        if "structure_specialist" in self.agents:
            structure_output = self.agents["structure_specialist"].run({
                "task": "create_novel_structure",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with structure results
            if "outline" in structure_output["output"]:
                project_state_manager.update_state({
                    "outline": structure_output["output"]["outline"]
                })
                
            # Store agent output
            agent_outputs["structure_specialist"] = structure_output
            
        # Run the plot development agent
        if "plot_development" in self.agents:
            plot_output = self.agents["plot_development"].run({
                "task": "develop_plot",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with plot results
            project_updates = {}
            if "plot_points" in plot_output["output"]:
                project_updates["plot_points"] = plot_output["output"]["plot_points"]
            if "themes" in plot_output["output"]:
                project_updates["themes"] = plot_output["output"]["themes"]
                
            project_state_manager.update_state(project_updates)
                
            # Store agent output
            agent_outputs["plot_development"] = plot_output
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.PLANNING,
            "phase_output": {
                "planning_completed": True
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_character_development(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run character development phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Run the character development agents
        agent_outputs = state.get("agent_outputs", {})
        
        if "character_psychology" in self.agents:
            character_output = self.agents["character_psychology"].run({
                "task": "develop_characters",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with character results
            if "characters" in character_output["output"]:
                for character in character_output["output"]["characters"]:
                    project_state_manager.add_character(character)
                
            # Store agent output
            agent_outputs["character_psychology"] = character_output
            
        # Run the character relationship mapper
        if "character_relationship" in self.agents:
            relationship_output = self.agents["character_relationship"].run({
                "task": "map_character_relationships",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with relationship results
            if "character_relationships" in relationship_output["output"]:
                for character_id, relationships in relationship_output["output"]["character_relationships"].items():
                    character = project_state_manager.get_character(character_id)
                    character["relationships"] = relationships
                    project_state_manager.update_character(character_id, character)
                
            # Store agent output
            agent_outputs["character_relationship"] = relationship_output
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.CHARACTER_DEVELOPMENT,
            "phase_output": {
                "character_development_completed": True
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_drafting(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run drafting phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.DRAFTING)
        
        # Get the outline
        outline = project_state_manager.get_state().get("outline", [])
        
        # Run the chapter writer agent for each outline item
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs.setdefault("chapter_writer", {})
        
        for i, outline_item in enumerate(outline):
            chapter_number = i + 1
            chapter_id = outline_item.get("id", str(uuid.uuid4()))
            
            # Run the chapter writer agent
            if "chapter_writer" in self.agents:
                chapter_output = self.agents["chapter_writer"].run({
                    "task": "write_chapter",
                    "project_state": project_state_manager.get_state(),
                    "chapter_number": chapter_number,
                    "outline_item": outline_item
                })
                
                # Update project with chapter results
                if "chapter" in chapter_output["output"]:
                    chapter_data = chapter_output["output"]["chapter"]
                    chapter_data["id"] = chapter_id
                    chapter_data["sequence_number"] = chapter_number
                    project_state_manager.add_chapter(chapter_data)
                    
                # Store agent output
                agent_outputs["chapter_writer"][chapter_id] = chapter_output
        
        # Increment draft number
        project_state_manager.increment_draft_number()
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.DRAFTING,
            "phase_output": {
                "drafting_completed": True,
                "draft_number": project_state_manager.get_state()["current_draft_number"]
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_revision(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run revision phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.REVISING)
        
        # Run the structural editor agent
        agent_outputs = state.get("agent_outputs", {})
        if "structural_editor" in self.agents:
            structural_output = self.agents["structural_editor"].run({
                "task": "review_structure",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with structural review results
            if "structure_feedback" in structural_output["output"]:
                project_state_manager.add_feedback({
                    "content": structural_output["output"]["structure_feedback"],
                    "source": "structural_editor",
                    "type": "structure"
                })
                
            # Store agent output
            agent_outputs["structural_editor"] = structural_output
            
        # Run the prose enhancement agent for each chapter
        agent_outputs.setdefault("prose_enhancement", {})
        for chapter in project_state_manager.get_chapters():
            if "prose_enhancement" in self.agents:
                prose_output = self.agents["prose_enhancement"].run({
                    "task": "enhance_prose",
                    "project_state": project_state_manager.get_state(),
                    "chapter_id": chapter["id"]
                })
                
                # Update chapter with enhanced prose
                if "enhanced_content" in prose_output["output"]:
                    project_state_manager.update_chapter(
                        chapter["id"], 
                        {"content": prose_output["output"]["enhanced_content"]}
                    )
                    
                # Store agent output
                agent_outputs["prose_enhancement"][chapter["id"]] = prose_output
        
        # Run the continuity checker agent
        if "continuity_checker" in self.agents:
            continuity_output = self.agents["continuity_checker"].run({
                "task": "check_continuity",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with continuity check results
            if "continuity_issues" in continuity_output["output"]:
                for issue in continuity_output["output"]["continuity_issues"]:
                    project_state_manager.add_feedback({
                        "content": issue["description"],
                        "source": "continuity_checker",
                        "type": "continuity",
                        "location": issue.get("location", {})
                    })
                    
            # Store agent output
            agent_outputs["continuity_checker"] = continuity_output
        
        # Increment draft number
        project_state_manager.increment_draft_number()
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.REVISION,
            "phase_output": {
                "revision_completed": True,
                "draft_number": project_state_manager.get_state()["current_draft_number"],
                "needs_more_revision": self._needs_more_revision(project_state_manager.get_state())
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_optimization(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run optimization phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.EDITING)
        
        # Run the emotional arc validator agent
        agent_outputs = state.get("agent_outputs", {})
        if "emotional_arc_validator" in self.agents:
            emotional_output = self.agents["emotional_arc_validator"].run({
                "task": "validate_emotional_arc",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with emotional arc validation results
            if "emotional_arc_feedback" in emotional_output["output"]:
                project_state_manager.add_feedback({
                    "content": emotional_output["output"]["emotional_arc_feedback"],
                    "source": "emotional_arc_validator",
                    "type": "emotional_arc"
                })
                
            # Store agent output
            agent_outputs["emotional_arc_validator"] = emotional_output
            
        # Run the hook optimization agent
        if "hook_optimization" in self.agents:
            hook_output = self.agents["hook_optimization"].run({
                "task": "optimize_hooks",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with hook optimization results
            if "optimized_hooks" in hook_output["output"]:
                for chapter_id, hook in hook_output["output"]["optimized_hooks"].items():
                    chapter = project_state_manager.get_chapter(chapter_id)
                    # Assuming the hook is at the beginning of the chapter
                    current_content = chapter["content"]
                    # Replace the first paragraph with the optimized hook
                    paragraphs = current_content.split("\n\n")
                    paragraphs[0] = hook
                    updated_content = "\n\n".join(paragraphs)
                    project_state_manager.update_chapter(
                        chapter_id, 
                        {"content": updated_content}
                    )
                    
            # Store agent output
            agent_outputs["hook_optimization"] = hook_output
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.OPTIMIZATION,
            "phase_output": {
                "optimization_completed": True
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _run_publication(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """
        Run publication phase.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        # Initialize or retrieve project state
        project_state_manager = ProjectState(state["project_state"])
        
        # Update project status
        project_state_manager.update_status(ProjectStatus.FINALIZING)
        
        # Run the blurb optimization agent
        agent_outputs = state.get("agent_outputs", {})
        if "blurb_optimization" in self.agents:
            blurb_output = self.agents["blurb_optimization"].run({
                "task": "create_blurb",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with blurb results
            if "blurb" in blurb_output["output"]:
                project_state_manager.update_state({
                    "blurb": blurb_output["output"]["blurb"]
                })
                
            # Store agent output
            agent_outputs["blurb_optimization"] = blurb_output
            
        # Run the formatting specialist agent
        if "formatting_specialist" in self.agents:
            formatting_output = self.agents["formatting_specialist"].run({
                "task": "format_manuscript",
                "project_state": project_state_manager.get_state()
            })
            
            # Update project with formatting results
            if "formatted_manuscript" in formatting_output["output"]:
                project_state_manager.update_state({
                    "formatted_manuscript": formatting_output["output"]["formatted_manuscript"]
                })
                
            # Store agent output
            agent_outputs["formatting_specialist"] = formatting_output
        
        # Update project status to completed
        project_state_manager.update_status(ProjectStatus.COMPLETED)
        
        # Return updated state
        return {
            "project_state": project_state_manager.get_state(),
            "current_phase": WorkflowPhase.PUBLICATION,
            "phase_output": {
                "publication_completed": True
            },
            "agent_outputs": agent_outputs,
            "errors": state.get("errors", [])
        }
    
    def _determine_next_phase(self, state: MasterWorkflowState) -> Union[str, END]:
        """
        Determine the next phase based on current state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Next phase or END
        """
        current_phase = state["current_phase"]
        phase_output = state.get("phase_output", {})
        
        # Check for errors
        if state.get("errors") and len(state["errors"]) > 0:
            return END
            
        # Phase-specific logic
        if current_phase == WorkflowPhase.INITIALIZATION:
            return WorkflowPhase.RESEARCH
        elif current_phase == WorkflowPhase.RESEARCH:
            return WorkflowPhase.PLANNING
        elif current_phase == WorkflowPhase.PLANNING:
            return WorkflowPhase.CHARACTER_DEVELOPMENT
        elif current_phase == WorkflowPhase.CHARACTER_DEVELOPMENT:
            return WorkflowPhase.DRAFTING
        elif current_phase == WorkflowPhase.DRAFTING:
            return WorkflowPhase.REVISION
        elif current_phase == WorkflowPhase.REVISION:
            # Check if more revision is needed
            if phase_output.get("needs_more_revision", False):
                return WorkflowPhase.REVISION
            else:
                return WorkflowPhase.OPTIMIZATION
        elif current_phase == WorkflowPhase.OPTIMIZATION:
            return WorkflowPhase.PUBLICATION
        else:
            return END
    
    def _needs_more_revision(self, project_state: Dict[str, Any]) -> bool:
        """
        Determine if more revision is needed.
        
        Args:
            project_state: The current project state
            
        Returns:
            True if more revision is needed, False otherwise
        """
        # Check if we've reached the maximum number of revisions
        max_revisions = self.config.get("max_revisions", 3)
        current_draft = project_state.get("current_draft_number", 0)
        
        if current_draft >= max_revisions:
            return False
            
        # Check if there are unresolved critical feedback items
        feedback = project_state.get("feedback", [])
        critical_feedback = [f for f in feedback if f.get("priority") == "critical" and not f.get("resolved", False)]
        
        if critical_feedback:
            return True
            
        # Additional revision criteria could be added here
        
        return False
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the entire workflow with given parameters.
        
        Args:
            initial_state: Initial state for the workflow
            
        Returns:
            Final workflow state
        """
        workflow_state: MasterWorkflowState = {
            "project_state": initial_state.get("project_state", {}),
            "current_phase": WorkflowPhase.INITIALIZATION,
            "phase_output": None,
            "agent_outputs": {},
            "errors": []
        }
        
        if "project_parameters" in initial_state:
            workflow_state["project_parameters"] = initial_state["project_parameters"]
            
        config = {"recursion_limit": self.config.get("max_iterations", 10)}
        result = self.graph.invoke(workflow_state, config=config)
        
        return result
