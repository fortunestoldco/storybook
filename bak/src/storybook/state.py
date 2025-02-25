"""State definitions for the Storybook workflow."""

from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, utc

from storybook.config import (
    StoryState,
    AgentRole,
    TeamType,
    BibleSectionType,
    FeedbackType,
    StoryStructure,
    OperationMode,
    UserRequest,
    ResearchItem,
    BibleSection,
    StoryBible,
    Character,
    PlotPoint,
    Setting,
    StoryOutline,
    StorySection,
    Feedback,
    PublishingMetadata,
    Story,
    WriterAssignment,
)


class AgentState(BaseModel):
    """Base state for all agents."""

    agent_id: str
    agent_role: AgentRole
    status: str = "idle"  # idle, working, waiting, error
    last_action: Optional[str] = None
    last_update: Optional[str] = None
    memory: Dict[str, Any] = Field(default_factory=dict)

    def update_status(self, new_status: str, action: Optional[str] = None):
        """Update agent status and last action."""
        import datetime

        self.status = new_status
        if action:
            self.last_action = action
        self.last_update = datetime.datetime.now().isoformat()


class ResearchAgentState(AgentState):
    """State for research agents."""

    research_topics: List[str] = Field(default_factory=list)
    completed_topics: List[str] = Field(default_factory=list)
    research_items: List[ResearchItem] = Field(default_factory=list)
    research_summary: Optional[str] = None
    current_task: Optional[str] = None
    search_queries: List[str] = Field(default_factory=list)

    def add_research_item(self, item: ResearchItem):
        """Add a research item to the agent's state."""
        self.research_items.append(item)

    def mark_topic_complete(self, topic: str):
        """Mark a research topic as complete."""
        if topic in self.research_topics and topic not in self.completed_topics:
            self.completed_topics.append(topic)


class WritingAgentState(AgentState):
    """State for writing agents."""

    current_section: Optional[str] = None
    assigned_sections: List[str] = Field(default_factory=list)
    completed_sections: List[str] = Field(default_factory.list)
    draft_content: Dict[str, str] = Field(default_factory=dict)
    revision_count: Dict[str, int] = Field(default_factory=dict)
    feedback_incorporated: List[str] = Field(default_factory=list)
    bible_sections_authored: List[str] = Field(default_factory=list)
    is_joint_llm: bool = False  # Whether this is a joint LLM writer
    collaborating_with: List[str] = Field(
        default_factory=list
    )  # Other writers collaborating on joint sections

    def add_draft_content(self, section: str, content: str):
        """Add or update draft content for a section."""
        self.draft_content[section] = content

    def mark_section_complete(self, section: str):
        """Mark a section as complete."""
        if section in self.assigned_sections and section not in self.completed_sections:
            self.completed_sections.append(section)

    def increment_revision(self, section: str):
        """Increment the revision count for a section."""
        if section not in self.revision_count:
            self.revision_count[section] = 0
        self.revision_count[section] += 1

    def is_section_assigned(self, section: str) -> bool:
        """Check if a section is assigned to this writer."""
        return section in self.assigned_sections


class JointWriterAgentState(WritingAgentState):
    """Special state for joint writer agents that collaborate on complex sections."""

    component_writers: List[str] = Field(
        default_factory=list
    )  # IDs of writers contributing to this joint entity
    section_contributions: Dict[str, Dict[str, str]] = Field(
        default_factory=dict
    )  # Section ID -> {writer_id: contribution}

    def add_contribution(self, section_id: str, writer_id: str, content: str):
        """Add a writer's contribution to a section."""
        if section_id not in self.section_contributions:
            self.section_contributions[section_id] = {}
        self.section_contributions[section_id][writer_id] = content

    def get_all_contributions(self, section_id: str) -> Dict[str, str]:
        """Get all writer contributions for a section."""
        return self.section_contributions.get(section_id, {})


class EditingAgentState(AgentState):
    """State for editing agents."""

    assigned_sections: List[str] = Field(default_factory=list)
    completed_sections: List[str] = Field(default_factory.list)
    current_section: Optional[str] = None
    edit_notes: Dict[str, List[str]] = Field(default_factory=dict)
    edit_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    def add_edit_note(self, section: str, note: str):
        """Add an editing note for a section."""
        if section not in self.edit_notes:
            self.edit_notes[section] = []
        self.edit_notes[section].append(note)

    def record_edit(self, section: str, edit_type: str, description: str):
        """Record an edit made to a section."""
        if section not in self.edit_history:
            self.edit_history[section] = []

        import datetime

        edit_record = {
            "type": edit_type,
            "description": description,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.edit_history[section].append(edit_record)


class PublishingAgentState(AgentState):
    """State for publishing agents."""

    publishing_metadata: Optional[PublishingMetadata] = None
    format_status: str = "not_started"  # not_started, in_progress, complete
    seo_status: str = "not_started"
    marketing_materials: Dict[str, str] = Field(default_factory=dict)
    publish_platforms: List[str] = Field(default_factory=list)
    publish_urls: Dict[str, str] = Field(default_factory.dict)

    def add_marketing_material(self, material_type: str, content: str):
        """Add marketing material for the story."""
        self.marketing_materials[material_type] = content

    def add_publish_url(self, platform: str, url: str):
        """Add a publishing URL for a platform."""
        self.publish_urls[platform] = url


class SupervisorAgentState(AgentState):
    """State for supervisor agents."""

    team_type: TeamType
    supervised_agents: List[str] = Field(default_factory.list)
    agent_assignments: Dict[str, List[str]] = Field(default_factory.dict)
    approvals: Dict[str, bool] = Field(default_factory.dict)
    pending_reviews: List[str] = Field(default_factory.list)
    completed_reviews: List[str] = Field(default_factory.list)
    feedback_given: Dict[str, List[str]] = Field(default_factory.dict)

    def assign_task(self, agent_id: str, task: str):
        """Assign a task to an agent."""
        if agent_id not in self.agent_assignments:
            self.agent_assignments[agent_id] = []
        self.agent_assignments[agent_id].append(task)

    def record_approval(self, item_id: str, approved: bool):
        """Record an approval decision for an item."""
        self.approvals[item_id] = approved

    def add_feedback(self, target_id: str, feedback_id: str):
        """Record feedback given to a target."""
        if target_id not in self.feedback_given:
            self.feedback_given[target_id] = []
        self.feedback_given[target_id].append(feedback_id)


class AuthorRelationsAgentState(AgentState):
    """State for the author relations agent."""

    current_session: Optional[str] = None
    session_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory.dict)
    brainstorm_topics: List[str] = Field(default_factory.list)
    user_preferences: Dict[str, Any] = Field(default_factory.dict)
    pending_feedback_requests: List[str] = Field(default_factory.list)
    received_feedback: List[Feedback] = Field(default_factory.list)

    def start_session(self, session_id: str, topic: str):
        """Start a new session with a topic."""
        self.current_session = session_id
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        if topic not in self.brainstorm_topics:
            self.brainstorm_topics.append(topic)

    def add_session_message(self, session_id: str, role: str, content: str):
        """Add a message to a session."""
        if session_id not in self.session_history:
            self.session_history[session_id] = []

        import datetime

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.session_history[session_id].append(message)

    def end_session(self, session_id: str):
        """End the current session."""
        if self.current_session == session_id:
            self.current_session = None


class HumanInLoopState(AgentState):
    """State for human-in-the-loop review."""

    pending_reviews: List[Dict[str, Any]] = Field(default_factory.list)
    completed_reviews: List[Dict[str, Any]] = Field(default_factory.list)
    current_review: Optional[Dict[str, Any]] = None
    review_deadline: Optional[str] = None

    def add_pending_review(self, review_data: Dict[str, Any]):
        """Add a pending review."""
        self.pending_reviews.append(review_data)

    def complete_review(self, review_id: str, decision: str, comments: Optional[str] = None):
        """Mark a review as complete with decision and comments."""
        for i, review in enumerate(self.pending_reviews):
            if review.get("id") == review_id:
                review["decision"] = decision
                review["comments"] = comments
                review["completed_at"] = datetime.datetime.now().isoformat()
                self.completed_reviews.append(review)
                self.pending_reviews.pop(i)
                break

        if self.current_review and self.current_review.get("id") == review_id:
            self.current_review = None


class StyleGuideEditorState(AgentState):
    """State for the style guide editor agent."""

    bible_sections: Dict[str, List[BibleSection]] = Field(default_factory.dict)
    current_section: Optional[str] = None
    revision_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory.dict)

    def add_bible_section(self, section: BibleSection):
        """Add a bible section to the agent's state."""
        section_type = section.section_type.value
        if section_type not in self.bible_sections:
            self.bible_sections[section_type] = []
        self.bible_sections[section_type].append(section)

    def record_revision(self, section_id: str, description: str, content: str):
        """Record a revision to a bible section."""
        if section_id not in self.revision_history:
            self.revision_history[section_id] = []

        import datetime

        revision = {
            "description": description,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.revision_history[section_id].append(revision)


class GraphState(BaseModel):
    """Overall state for the Storybook workflow graph."""

    story_id: Optional[str] = None
    user_request: Optional[UserRequest] = None
    story: Optional[Story] = None
    current_state: StoryState = StoryState.INITIATED
    operation_mode: OperationMode = OperationMode.CREATE

    # Bible tracking
    bible: Optional[StoryBible] = None
    bible_status: Dict[str, str] = Field(default_factory.dict)

    # Team states
    research_team: Dict[str, ResearchAgentState] = Field(default_factory.dict)
    writing_team: Dict[str, WritingAgentState] = Field(default_factory.dict)
    joint_writers: Dict[str, JointWriterAgentState] = Field(default_factory.dict)
    editing_team: Dict[str, EditingAgentState] = Field(default_factory.dict)
    publishing_team: Dict[str, PublishingAgentState] = Field(default_factory.dict)
    supervisors: Dict[str, SupervisorAgentState] = Field(default_factory.dict)

    # Special agents
    author_relations: Dict[str, AuthorRelationsAgentState] = Field(default_factory.dict)
    human_in_loop: Dict[str, HumanInLoopState] = Field(default_factory.dict)
    style_guide_editor: Dict[str, StyleGuideEditorState] = Field(default_factory.dict)

    # Communication and coordination
    messages: List[Dict[str, Any]] = Field(default_factory.list)
    tasks: Dict[str, Dict[str, Any]] = Field(default_factory.dict)

    # Story structure tracking
    story_structure: StoryStructure = StoryStructure.THREE_ACT
    structure_template: Dict[str, Any] = Field(default_factory.dict)

    # Writer assignment tracking
    writer_assignments: List[WriterAssignment] = Field(default_factory.list)
    num_writers: int = 1
    use_joint_llm: bool = False

    # Workflow tracking
    phase_completion: Dict[StoryState, bool] = Field(default_factory.dict)
    current_phase_progress: float = 0.0  # 0.0 to 1.0

    # Import/Edit mode tracking
    imported_content: Optional[str] = None
    sections_to_edit: List[str] = Field(default_factory.list)

    # Human interaction tracking
    awaiting_human_input: bool = False
    human_input_type: Optional[str] = None
    human_feedback: List[Dict[str, Any]] = Field(default_factory.list)

    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory.list)
    retries: Dict[str, int] = Field(default_factory.dict)

    # Timestamps
    created_at: Optional[str] = None
    last_updated: Optional[str] = None
    deadline: Optional[str] = None

    def add_message(
        self, sender: str, recipient: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the communication log."""
        if metadata is None:
            metadata = {}

        import datetime

        timestamp = datetime.datetime.now().isoformat()

        message = {
            "id": f"msg_{len(self.messages) + 1}",
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata,
        }

        self.messages.append(message)
        self.last_updated = timestamp
        return message

    def add_task(
        self,
        task_id: str,
        agent_id: str,
        task_type: str,
        description: str,
        status: str = "assigned",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add a task to the task tracking system."""
        if data is None:
            data = {}

        import datetime

        timestamp = datetime.datetime.now().isoformat()

        task = {
            "id": task_id,
            "agent_id": agent_id,
            "type": task_type,
            "description": description,
            "status": status,
            "data": data,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        self.tasks[task_id] = task
        self.last_updated = timestamp
        return task

    def update_task_status(
        self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None
    ):
        """Update the status of a task."""
        if task_id not in self.tasks:
            return None

        import datetime

        timestamp = datetime.datetime.now().isoformat()

        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["updated_at"] = timestamp

        if result:
            self.tasks[task_id]["result"] = result

        self.last_updated = timestamp
        return self.tasks[task_id]

    def add_error(self, source: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """Log an error in the workflow."""
        if details is None:
            details = {}

        import datetime

        timestamp = datetime.datetime.now().isoformat()

        error = {
            "source": source,
            "error_message": error_message,
            "timestamp": timestamp,
            "details": details,
        }

        self.errors.append(error)
        self.last_updated = timestamp
        return error

    def increment_retry(self, key: str) -> int:
        """Increment retry count for a key and return the new count."""
        if key not in self.retries:
            self.retries[key] = 0
        self.retries[key] += 1
        return self.retries[key]

    def update_story_state(self, new_state: StoryState):
        """Update the story state and reset phase progress."""
        self.current_state = new_state
        self.current_phase_progress = 0.0

        if self.story:
            self.story.state = new_state

        import datetime

        self.last_updated = datetime.datetime.now().isoformat()

    def update_phase_progress(self, progress: float):
        """Update the progress of the current phase (0.0 to 1.0)."""
        self.current_phase_progress = min(1.0, max(0.0, progress))

        import datetime

        self.last_updated = datetime.datetime.now().isoformat()

    def mark_phase_complete(self, phase: StoryState):
        """Mark a workflow phase as complete."""
        self.phase_completion[phase] = True

        if phase == self.current_state:
            self.current_phase_progress = 1.0

        import datetime

        self.last_updated = datetime.datetime.now().isoformat()

    def request_human_input(self, input_type: str, data: Optional[Dict[str, Any]] = None):
        """Mark the workflow as awaiting human input of a specific type."""
        self.awaiting_human_input = True
        self.human_input_type = input_type

        # Add a task for human in the loop
        human_agent_id = next(iter(self.human_in_loop.keys()), "human_agent")

        task_id = f"human_task_{len(self.tasks) + 1}"
        self.add_task(
            task_id=task_id,
            agent_id=human_agent_id,
            task_type=input_type,
            description=f"Human input required: {input_type}",
            status="pending",
            data=data or {},
        )

        # Add to human agent's pending reviews
        if human_agent_id in self.human_in_loop:
            human_state = self.human_in_loop[human_agent_id]
            review_data = {
                "id": task_id,
                "type": input_type,
                "data": data or {},
                "created_at": datetime.datetime.now().isoformat(),
            }
            human_state.add_pending_review(review_data)

        self.last_updated = datetime.datetime.now().isoformat()

def process_human_input(self, input_id: str, response: Dict[str, Any]):
        """Process human input and update the workflow state."""
        self.awaiting_human_input = False

        feedback = {
            "id": input_id,
            "type": self.human_input_type,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.human_feedback.append(feedback)
        self.human_input_type = None

        if input_id in self.tasks:
            self.update_task_status(task_id=input_id, status="completed", result=response)

        for human_agent_id, human_state in self.human_in_loop.items():
            human_state.complete_review(
                review_id=input_id,
                decision=response.get("decision", "completed"),
                comments=response.get("comments"),
            )

        self.last_updated = datetime.datetime.now().isoformat()
        return feedback

    def assign_writer_to_sections(
        self, writer_id: str, section_ids: List[str], joint_llm: bool = False
    ):
        """Assign a writer to specific story sections."""
        if self.story:
            self.story.assign_writer(writer_id, section_ids, joint_llm)

        for assignment in self.writer_assignments:
            if assignment.writer_id == writer_id:
                assignment.sections.extend([s for s in section_ids if s not in assignment.sections])
                assignment.joint_llm = joint_llm
                return

        self.writer_assignments.append(
            WriterAssignment(writer_id=writer_id, sections=section_ids, joint_llm=joint_llm)
        )

        if writer_id in self.writing_team:
            writer_state = self.writing_team[writer_id]
            writer_state.assigned_sections.extend(
                [s for s in section_ids if s not in writer_state.assigned_sections]
            )
            writer_state.is_joint_llm = joint_llm

    def get_sections_for_writer(self, writer_id: str) -> List[str]:
        """Get the sections assigned to a specific writer."""
        for assignment in self.writer_assignments:
            if assignment.writer_id == writer_id:
                return assignment.sections
        return []

    def is_joint_llm_writer(self, writer_id: str) -> bool:
        """Check if a writer is assigned to use joint LLM."""
        for assignment in self.writer_assignments:
            if assignment.writer_id == writer_id:
                return assignment.joint_llm
        return False

    def create_section_from_template(
        self, act_index: int, component_index: int, title: Optional[str] = None
    ) -> StorySection:
        """Create a story section based on the structure template."""
        import datetime
        from agents.utils import generate_id

        if not self.structure_template or "acts" not in self.structure_template:
            raise ValueError("No valid structure template available")

        if act_index >= len(self.structure_template["acts"]):
            raise ValueError(f"Act index {act_index} out of range")

        act = self.structure_template["acts"][act_index]

        if component_index >= len(act.get("components", [])):
            raise ValueError(f"Component index {component_index} out of range")

        component = act["components"][component_index]

        section_id = generate_id("section")
        sequence = (act_index * 100) + component_index
        section_title = title or component.get("name", f"Section {sequence}")

        return StorySection(
            id=section_id,
            title=section_title,
            content="",
            sequence=sequence,
            act=act.get("name", f"Act {act_index + 1}"),
            status="not_started",
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
        )       
