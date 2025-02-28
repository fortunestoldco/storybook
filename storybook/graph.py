from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.message import ToolCall, ToolResponse
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from uuid import uuid4

from storybook.state import State, AgentOutput
from storybook.agents import (
    AuthorRelationsAgent, OrchestratorAgent, 
    ResearchTeamLead, WritingTeamLead, EditorialTeamLead,
    Researcher, Writer, Editor
)
from storybook.config import Configuration

# Define input schema with Enums
class SubmissionType(str, Enum):
    NEW = "New"
    EXISTING = "Existing"

class ModelType(str, Enum):
    GPT_4 = "GPT-4"
    CLAUDE_3 = "Claude 3"
    GEMINI = "Gemini"
    LLAMA_3 = "Llama 3"

class CustomModelType(str, Enum):
    NONE = "None"
    FICTION_SPECIALIZED = "Fiction Specialized"
    TECHNICAL_WRITING = "Technical Writing"
    CREATIVE_PLUS = "Creative Plus"

class InputState(BaseModel):
    submission_type: SubmissionType
    title: Optional[str] = None
    manuscript: Optional[str] = None
    model: ModelType
    custom_model: Optional[CustomModelType] = CustomModelType.NONE
    project_id: Optional[str] = None

class TeamState(BaseModel):
    team_id: str
    members: List[str]
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    current_task: Optional[Dict[str, Any]] = None
    completed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    reports: List[Dict[str, Any]] = Field(default_factory=list)

class ProjectState(BaseModel):
    project_id: str
    title: str
    manuscript: str
    status: str = "initiated"
    draft_number: int = 0
    brainstorm_notes: Optional[str] = None
    research_team: Optional[TeamState] = None
    writing_team: Optional[TeamState] = None
    editorial_team: Optional[TeamState] = None
    
class EnhancedState(State):
    submission: InputState
    project: Optional[ProjectState] = None
    human_in_loop: bool = False
    current_team: Optional[str] = None
    current_phase: str = "start"
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)

# Node functions
async def initialize_workflow(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Initialize the workflow based on submission type."""
    if state.submission.submission_type == SubmissionType.NEW:
        # New submission path
        return {"current_phase": "author_relations_chat"}
    else:
        # Existing submission - retrieve project
        if not state.submission.project_id:
            return {"current_phase": "error", "error": "Project ID required for existing submissions"}
        
        # Mock retrieval of existing project
        # In a real implementation, this would fetch from a database
        project_id = state.submission.project_id
        return {
            "project": ProjectState(
                project_id=project_id,
                title=state.submission.title or "Retrieved Title",
                manuscript=state.submission.manuscript or "Retrieved Manuscript",
                status="in_progress"
            ),
            "current_phase": "orchestrator_resume"
        }

async def author_relations_chat(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Human-in-loop chat with Author Relations Agent."""
    agent = AuthorRelationsAgent(config)
    
    if not state.human_in_loop:
        # First time entering this node - start the conversation
        welcome_message = await agent.start_conversation(state.submission)
        return {
            "human_in_loop": True,
            "chat_history": [{"role": "system", "content": welcome_message}],
            "current_phase": "author_relations_chat"  # Stay in this node
        }
    
    # Check if user wants to end the brainstorm
    last_message = state.chat_history[-1] if state.chat_history else None
    if last_message and last_message.get("role") == "user" and "end brainstorm" in last_message.get("content", "").lower():
        # Generate ticket number and summary
        project_id = f"PRJ-{uuid4().hex[:8].upper()}"
        summary = await agent.summarize_brainstorm(state.chat_history)
        
        # Prepare ticket info for user
        ticket_info = f"Thank you for your input! Your project ticket number is {project_id}."
        
        return {
            "human_in_loop": False,
            "chat_history": state.chat_history + [{"role": "system", "content": ticket_info}],
            "project": ProjectState(
                project_id=project_id,
                title=state.submission.title or "Untitled Project",
                manuscript=state.submission.manuscript or "",
                brainstorm_notes=summary
            ),
            "current_phase": "orchestrator_initialize"
        }
    
    # Continue the conversation
    response = await agent.continue_conversation(state.chat_history)
    updated_history = state.chat_history + [{"role": "system", "content": response}]
    
    return {
        "human_in_loop": True,
        "chat_history": updated_history,
        "current_phase": "author_relations_chat"  # Stay in this node
    }

async def orchestrator_initialize(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Orchestrator initializes the project and teams."""
    agent = OrchestratorAgent(config)
    
    # Initialize the project in tracker
    project_update = await agent.initialize_project(state.project)
    
    # Setup teams
    research_team = TeamState(
        team_id=f"RT-{uuid4().hex[:6].upper()}",
        members=["senior_researcher", "domain_specialist", "data_analyst"]
    )
    
    writing_team = TeamState(
        team_id=f"WT-{uuid4().hex[:6].upper()}",
        members=["lead_writer", "creative_writer", "narrative_designer"]
    )
    
    editorial_team = TeamState(
        team_id=f"ET-{uuid4().hex[:6].upper()}",
        members=["editor_in_chief", "copy_editor", "quality_reviewer"]
    )
    
    # Assign initial tasks
    research_tasks = await agent.assign_research_tasks(state.project)
    writing_tasks = await agent.assign_ideation_tasks(state.project)
    
    research_team.tasks = research_tasks
    research_team.current_task = research_tasks[0] if research_tasks else None
    
    writing_team.tasks = writing_tasks
    writing_team.current_task = writing_tasks[0] if writing_tasks else None
    
    updated_project = state.project.copy()
    updated_project.status = "in_progress"
    updated_project.research_team = research_team
    updated_project.writing_team = writing_team
    updated_project.editorial_team = editorial_team
    
    return {
        "project": updated_project,
        "current_phase": "research_team_work",
        "current_team": "research"
    }

async def orchestrator_resume(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Orchestrator resumes an existing project."""
    agent = OrchestratorAgent(config)
    
    # Determine the current state of the project and what should happen next
    next_phase = await agent.determine_next_phase(state.project)
    
    return {
        "current_phase": next_phase
    }

async def research_team_work(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Research team performs their assigned tasks."""
    team_lead = ResearchTeamLead(config)
    researcher = Researcher(config)
    
    research_team = state.project.research_team
    if not research_team or not research_team.current_task:
        return {"current_phase": "orchestrator_update", "error": "No research tasks assigned"}
    
    # Perform current research task
    task_result = await researcher.perform_task(
        research_team.current_task, 
        state.project
    )
    
    # Update team state
    updated_team = research_team.copy()
    if updated_team.current_task:
        completed_task = updated_team.current_task.copy()
        completed_task["result"] = task_result
        completed_task["status"] = "completed"
        completed_task["completion_time"] = datetime.now().isoformat()
        
        updated_team.completed_tasks.append(completed_task)
        
        # Prepare research report
        report = await team_lead.prepare_report(completed_task, state.project)
        updated_team.reports.append({
            "id": f"RR-{uuid4().hex[:6].upper()}",
            "task_id": completed_task.get("id"),
            "content": report,
            "timestamp": datetime.now().isoformat()
        })
        
        # Move to next task if available
        if updated_team.tasks:
            updated_team.tasks.pop(0)  # Remove the completed task from the queue
            updated_team.current_task = updated_team.tasks[0] if updated_team.tasks else None
    
    # Update project with the updated team
    updated_project = state.project.copy()
    updated_project.research_team = updated_team
    
    # Determine next phase
    if updated_team.current_task:
        # More research tasks to do
        return {
            "project": updated_project,
            "current_phase": "research_team_work"  # Continue research
        }
    else:
        # Research complete for now
        return {
            "project": updated_project,
            "current_phase": "orchestrator_update",
            "current_team": "research"
        }

async def writing_team_work(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Writing team performs their assigned tasks."""
    team_lead = WritingTeamLead(config)
    writer = Writer(config)
    
    writing_team = state.project.writing_team
    if not writing_team or not writing_team.current_task:
        return {"current_phase": "orchestrator_update", "error": "No writing tasks assigned"}
    
    # Get latest research reports that might be useful
    research_reports = []
    if state.project.research_team and state.project.research_team.reports:
        research_reports = state.project.research_team.reports
    
    # Perform current writing task
    task_result = await writer.perform_task(
        writing_team.current_task,
        state.project,
        research_reports
    )
    
    # Update team state
    updated_team = writing_team.copy()
    if updated_team.current_task:
        completed_task = updated_team.current_task.copy()
        completed_task["result"] = task_result
        completed_task["status"] = "completed"
        completed_task["completion_time"] = datetime.now().isoformat()
        
        updated_team.completed_tasks.append(completed_task)
        
        # Prepare writing output
        report = await team_lead.prepare_output(completed_task, state.project)
        updated_team.reports.append({
            "id": f"WR-{uuid4().hex[:6].upper()}",
            "task_id": completed_task.get("id"),
            "content": report,
            "timestamp": datetime.now().isoformat()
        })
        
        # Move to next task if available
        if updated_team.tasks:
            updated_team.tasks.pop(0)  # Remove the completed task from the queue
            updated_team.current_task = updated_team.tasks[0] if updated_team.tasks else None
    
    # Update project with the updated team
    updated_project = state.project.copy()
    updated_project.writing_team = updated_team
    
    # If the writing team completed a draft, update the manuscript
    if any(task.get("type") == "complete_draft" for task in updated_team.completed_tasks[-1:]):
        updated_project.manuscript = task_result
        updated_project.draft_number += 1
    
    # Determine next phase
    if updated_team.current_task:
        # More writing tasks to do
        return {
            "project": updated_project,
            "current_phase": "writing_team_work"  # Continue writing
        }
    else:
        # Writing phase complete
        return {
            "project": updated_project,
            "current_phase": "orchestrator_update",
            "current_team": "writing"
        }

async def editorial_team_work(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Editorial team reviews and provides feedback."""
    team_lead = EditorialTeamLead(config)
    editor = Editor(config)
    
    editorial_team = state.project.editorial_team
    if not editorial_team:
        return {"current_phase": "orchestrator_update", "error": "Editorial team not initialized"}
    
    # Assign review task if not assigned
    if not editorial_team.current_task:
        review_task = await team_lead.create_review_task(state.project)
        editorial_team.current_task = review_task
    
    # Perform editorial review
    feedback = await editor.review_manuscript(
        state.project.manuscript,
        state.project.draft_number
    )
    
    # Update team state
    updated_team = editorial_team.copy()
    completed_task = updated_team.current_task.copy()
    completed_task["result"] = feedback
    completed_task["status"] = "completed"
    completed_task["completion_time"] = datetime.now().isoformat()
    
    updated_team.completed_tasks.append(completed_task)
    updated_team.current_task = None
    
    # Prepare editorial report
    report = await team_lead.prepare_feedback(completed_task, state.project)
    updated_team.reports.append({
        "id": f"ER-{uuid4().hex[:6].upper()}",
        "task_id": completed_task.get("id"),
        "content": report,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update project with the updated team
    updated_project = state.project.copy()
    updated_project.editorial_team = updated_team
    
    return {
        "project": updated_project,
        "current_phase": "orchestrator_update",
        "current_team": "editorial"
    }

async def orchestrator_update(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Orchestrator updates project status and determines next steps."""
    agent = OrchestratorAgent(config)
    
    current_team = state.current_team
    
    if current_team == "research" and state.project.research_team and state.project.research_team.reports:
        # Research completed a task, update project and pass to writing if they have a report
        latest_report = state.project.research_team.reports[-1]
        
        # Determine if writing team should receive this research
        if state.project.writing_team and state.project.writing_team.current_task:
            # Assign new writing tasks based on research
            writing_tasks = await agent.update_writing_tasks(
                state.project, 
                latest_report
            )
            
            updated_project = state.project.copy()
            updated_project.writing_team.tasks.extend(writing_tasks)
            if not updated_project.writing_team.current_task and writing_tasks:
                updated_project.writing_team.current_task = writing_tasks[0]
            
            # Continue research in parallel if there are tasks
            if updated_project.research_team.current_task:
                return {
                    "project": updated_project,
                    "current_phase": "research_team_work",
                    "current_team": "research"
                }
            else:
                return {
                    "project": updated_project,
                    "current_phase": "writing_team_work",
                    "current_team": "writing"
                }
    
    elif current_team == "writing":
        # Writing team completed work, determine if it needs editorial review
        if state.project.draft_number > 0:
            return {
                "current_phase": "editorial_team_work",
                "current_team": "editorial"
            }
        else:
            # Continue with more writing tasks if available
            if state.project.writing_team and state.project.writing_team.current_task:
                return {
                    "current_phase": "writing_team_work",
                    "current_team": "writing"
                }
            
            # If no more writing tasks but research is still working, wait for research
            if state.project.research_team and state.project.research_team.current_task:
                return {
                    "current_phase": "research_team_work",
                    "current_team": "research"
                }
    
    elif current_team == "editorial":
        # Editorial provided feedback, check draft number
        if state.project.draft_number >= 3:
            # Three drafts complete, time for final author review
            return {
                "current_phase": "final_author_review",
                "current_team": None
            }
        else:
            # Assign revision tasks to writers based on editorial feedback
            latest_feedback = state.project.editorial_team.reports[-1]
            revision_tasks = await agent.create_revision_tasks(state.project, latest_feedback)
            
            updated_project = state.project.copy()
            updated_project.writing_team.tasks = revision_tasks
            updated_project.writing_team.current_task = revision_tasks[0] if revision_tasks else None
            
            return {
                "project": updated_project,
                "current_phase": "writing_team_work",
                "current_team": "writing"
            }
    
    # Default fallback - determine next step based on project state
    next_phase = await agent.determine_next_phase(state.project)
    
    return {
        "current_phase": next_phase
    }

async def final_author_review(state: EnhancedState, config: RunnableConfig) -> Dict[str, Any]:
    """Final human-in-loop review with the author."""
    agent = AuthorRelationsAgent(config)
    
    if not state.human_in_loop:
        # First time entering this node - present the manuscript
        presentation = await agent.present_final_manuscript(state.project)
        return {
            "human_in_loop": True,
            "chat_history": [{"role": "system", "content": presentation}],
            "current_phase": "final_author_review"  # Stay in this node
        }
    
    # Check if user wants to provide notes or complete the project
    last_message = state.chat_history[-1] if state.chat_history else None
    if last_message and last_message.get("role") == "user":
        user_input = last_message.get("content", "").lower()
        
        if "complete" in user_input or "commit" in user_input or "finalize" in user_input:
            # User wants to complete the project
            completion_message = await agent.complete_project(state.project)
            
            updated_project = state.project.copy()
            updated_project.status = "completed"
            
            return {
                "human_in_loop": False,
                "chat_history": state.chat_history + [{"role": "system", "content": completion_message}],
                "project": updated_project,
                "current_phase": "complete"
            }
        elif "revise" in user_input or "note" in user_input or "change" in user_input:
            # User wants to provide revision notes
            confirmation = await agent.record_revision_notes(state.chat_history)
            
            # Create a new revision task for the writers
            updated_project = state.project.copy()
            revision_task = {
                "id": f"REV-{uuid4().hex[:6].upper()}",
                "type": "author_revision",
                "description": "Address author's feedback and revise manuscript",
                "notes": last_message.get("content"),
                "status": "assigned",
                "assignment_time": datetime.now().isoformat()
            }
            
            if not updated_project.writing_team.tasks:
                updated_project.writing_team.tasks = []
            
            updated_project.writing_team.tasks.append(revision_task)
            updated_project.writing_team.current_task = revision_task
            
            return {
                "human_in_loop": False,
                "chat_history": state.chat_history + [{"role": "system", "content": confirmation}],
                "project": updated_project,
                "current_phase": "writing_team_work",
                "current_team": "writing"
            }
    
    # Continue the conversation
    response = await agent.continue_final_review(state.chat_history)
    updated_history = state.chat_history + [{"role": "system", "content": response}]
    
    return {
        "human_in_loop": True,
        "chat_history": updated_history,
        "current_phase": "final_author_review"  # Stay in this node
    }

def should_end(state: EnhancedState) -> bool:
    """Determine if the workflow should end."""
    return state.current_phase in ["complete", "error"]

def build_storybook(config: RunnableConfig) -> StateGraph:
    """Build and return the hierarchical multi-agent team graph."""
    # Initialize workflow with new input schema
    builder = StateGraph(EnhancedState, config_schema=Configuration)

    # Add nodes
    builder.add_node("initialize_workflow", initialize_workflow)
    builder.add_node("author_relations_chat", author_relations_chat)
    builder.add_node("orchestrator_initialize", orchestrator_initialize)
    builder.add_node("orchestrator_resume", orchestrator_resume)
    builder.add_node("orchestrator_update", orchestrator_update)
    builder.add_node("research_team_work", research_team_work)
    builder.add_node("writing_team_work", writing_team_work)
    builder.add_node("editorial_team_work", editorial_team_work)
    builder.add_node("final_author_review", final_author_review)

    # Add conditional edges based on the current phase
    builder.add_conditional_edges(
        "initialize_workflow",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "author_relations_chat",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "orchestrator_initialize",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "orchestrator_resume",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "orchestrator_update",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "research_team_work",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "writing_team_work",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "editorial_team_work",
        lambda state: state.current_phase
    )
    
    builder.add_conditional_edges(
        "final_author_review",
        lambda state: state.current_phase
    )

    # Add end condition
    builder.add_edge_filter(should_end, END)

    # Set entry point
    builder.set_entry_point("initialize_workflow")

    # Compile and return graph
    graph = builder.compile()
    graph.name = "HierarchicalMultiAgentTeamGraph"
    return graph

# Export the graph builder function
__all__ = ["build_storybook"]
