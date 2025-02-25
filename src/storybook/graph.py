import os
import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union, Literal
import uuid
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.twilio import TwilioAPIWrapper

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
    STORY_STRUCTURES,
    USE_GPU,
    USE_OLLAMA,
)

from storybook.state import (
    GraphState,
    AgentState,
    ResearchAgentState,
    WritingAgentState,
    JointWriterAgentState,
    EditingAgentState,
    SupervisorAgentState,
    AuthorRelationsAgentState,
    HumanInLoopState,
    StyleGuideEditorState,
)

from storybook.prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    EDITOR_SYSTEM_PROMPT,
    WRITING_SUPERVISOR_SYSTEM_PROMPT,
    AUTHOR_RELATIONS_SYSTEM_PROMPT,
    HUMAN_IN_LOOP_SYSTEM_PROMPT,
    STYLE_GUIDE_EDITOR_SYSTEM_PROMPT,
    RESEARCH_TASK_PROMPT,
    WRITING_TASK_PROMPT,
    EDITING_TASK_PROMPT,
    REVIEW_TASK_PROMPT,
    BIBLE_UPDATE_PROMPT,
    BRAINSTORM_SESSION_PROMPT,
    HUMAN_REVIEW_PROMPT,
)

from storybook.tools import ToolsService

from storybook.utils import (
    generate_id,
    current_timestamp,
    extract_json_from_text,
    format_message_history,
    clean_and_format_text,
    create_task_id,
    parse_feedback,
    format_agent_response,
    validate_story_structure,
    prepare_human_review_prompt,
    format_brainstorm_session,
    get_story_structure_template,
    create_model_instance,
    create_section_structure_from_template,
    distribute_sections_to_writers,
)

# Initialize tool executor
tool_executor = ToolExecutor()

# Setup MongoDB Atlas Vector Store
vector_store = MongoDBAtlasVectorSearch(
    database="story_db",
    collection="vector_store",
    uri="your_mongodb_atlas_uri",
)

# Function to create LLM based on GPU availability
def create_llm(model_name: str):
    if os.getenv("GPU", "false").lower() == "true":
        return create_model_instance(model_name, use_local=True)
    else:
        return Replicate(
            model=f"replicate/{model_name}:latest",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )

# Twilio API Wrapper for notifications
twilio = TwilioAPIWrapper()

def notify_human_in_loop(message: str):
    to_number = os.getenv("TWILIO_TO_NUMBER")
    if to_number:
        twilio.run(message, to_number)
# Agent Factory Functions
def create_researcher_agent(agent_id: str) -> Any:
    """Create a researcher agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("meta-llama-3-8b-instruct")

    prompt = create_openai_tools_agent(
        llm=llm, tools=RESEARCH_TOOLS, system_message=RESEARCHER_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=RESEARCH_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_research_supervisor_agent(agent_id: str) -> Any:
    """Create a research supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("supervisor")

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        system_message=RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    )

    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
    )


def create_writer_agent(agent_id: str) -> Any:
    """Create a writer agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("writing")

    prompt = create_openai_tools_agent(
        llm=llm, tools=WRITING_TOOLS, system_message=WRITER_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=WRITING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_joint_writer_agent(agent_id: str, component_writer_ids: List[str] = None) -> Any:
    """Create a joint writer agent that combines the power of multiple models."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("writing")

    # Enhanced system prompt for joint writing
    joint_writer_prompt = f"""
    {WRITER_SYSTEM_PROMPT}
    
    As a joint writer, you are responsible for handling particularly complex or important story sections that
    require exceptional attention to detail and creative skill. You may need to integrate contributions from
    multiple individual writers to create a cohesive, high-quality section.
    
    Focus on:
    - Ensuring narrative continuity across writers' contributions
    - Maintaining consistent character voices and relationships
    - Creating seamless transitions between ideas and scenes
    - Elevating the overall literary quality
    
    Your work represents the highest standard of storytelling for this project.
    """

    prompt = create_openai_tools_agent(
        llm=llm, tools=WRITING_TOOLS, system_message=joint_writer_prompt
    )

    return AgentExecutor(
        agent=prompt, tools=WRITING_TOOLS, verbose=True, handle_parsing_errors=True
    )
def create_editor_agent(agent_id: str) -> Any:
    """Create an editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("writing")

    prompt = create_openai_tools_agent(
        llm=llm, tools=EDITING_TOOLS, system_message=EDITOR_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=EDITING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_writing_supervisor_agent(agent_id: str) -> Any:
    """Create a writing supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("supervisor")

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        system_message=WRITING_SUPERVISOR_SYSTEM_PROMPT,
    )

    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )


def create_author_relations_agent(agent_id: str) -> Any:
    """Create an author relations agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("author_relations")

    prompt = create_openai_tools_agent(
        llm=llm, tools=AUTHOR_RELATIONS_TOOLS, system_message=AUTHOR_RELATIONS_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=AUTHOR_RELATIONS_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_style_guide_editor_agent(agent_id: str) -> Any:
    """Create a style guide editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_llm("writing")

    prompt = create_openai_tools_agent(
        llm=llm, tools=BIBLE_EDITOR_TOOLS, system_message=STYLE_GUIDE_EDITOR_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=BIBLE_EDITOR_TOOLS, verbose=True, handle_parsing_errors=True
    )
# LangGraph Node Functions
def initialize_workflow(state: GraphState) -> GraphState:
    """Initialize the workflow with agents and initial state based on operation mode."""
    new_state = state.copy()

    # Set creation timestamp if not set
    if not new_state.created_at:
        new_state.created_at = current_timestamp()
        new_state.last_updated = current_timestamp()

    # Create a unique story ID if not provided
    if not new_state.story_id:
        new_state.story_id = generate_id("story")

    # Set operation mode if not set
    if not new_state.operation_mode and new_state.user_request:
        new_state.operation_mode = new_state.user_request.operation_mode or OperationMode.CREATE

    # Set number of writers from request
    num_writers = 1
    use_joint_llm = False
    if new_state.user_request:
        num_writers = new_state.user_request.num_writers or 1
        use_joint_llm = new_state.user_request.use_joint_llm or False

    new_state.num_writers = max(1, min(5, num_writers))  # Limit between 1-5 writers
    new_state.use_joint_llm = use_joint_llm

    # Set story structure
    if new_state.user_request and new_state.user_request.story_structure:
        new_state.story_structure = new_state.user_request.story_structure
        new_state.structure_template = get_story_structure_template(new_state.story_structure)
    else:
        new_state.story_structure = StoryStructure.THREE_ACT
        new_state.structure_template = get_story_structure_template(StoryStructure.THREE_ACT)

    # Initialize teams if they don't exist
    # Research team
    if not new_state.research_team:
        researcher_id = generate_id("researcher")
        new_state.research_team[researcher_id] = ResearchAgentState(
            agent_id=researcher_id, agent_role=AgentRole.RESEARCHER, status="idle"
        )

    # Research supervisor
    if not any(
        sup.agent_role == AgentRole.RESEARCH_SUPERVISOR for sup in new_state.supervisors.values()
    ):
        supervisor_id = generate_id("rsup")
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.RESEARCH_SUPERVISOR,
            team_type=TeamType.RESEARCH,
            status="idle",
            supervised_agents=list(new_state.research_team.keys()),
        )

    # Writing team - create multiple writer agents based on num_writers
    if not new_state.writing_team:
        for i in range(new_state.num_writers):
            writer_id = generate_id(f"writer_{i+1}")
            new_state.writing_team[writer_id] = WritingAgentState(
                agent_id=writer_id, agent_role=AgentRole.WRITER, status="idle"
            )

        # Create joint writer if enabled
        if new_state.use_joint_llm:
            joint_writer_id = generate_id("joint_writer")
            new_state.joint_writers[joint_writer_id] = JointWriterAgentState(
                agent_id=joint_writer_id,
                agent_role=AgentRole.JOINT_WRITER,
                status="idle",
                component_writers=list(new_state.writing_team.keys()),
                is_joint_llm=True,
            )

    # Editing team
    if not new_state.editing_team:
        editor_id = generate_id("editor")
        new_state.editing_team[editor_id] = EditingAgentState(
            agent_id=editor_id, agent_role=AgentRole.EDITOR, status="idle"
        )

    # Writing supervisor
    if not any(
        sup.agent_role == AgentRole.WRITING_SUPERVISOR for sup in new_state.supervisors.values()
    ):
        supervisor_id = generate_id("wsup")
        writer_ids = list(new_state.writing_team.keys())
        joint_writer_ids = list(new_state.joint_writers.keys())
        editor_ids = list(new_state.editing_team.keys())

        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.WRITING_SUPERVISOR,
            team_type=TeamType.WRITING,
            status="idle",
            supervised_agents=writer_ids + joint_writer_ids + editor_ids,
        )

    # Special agents
    # Author relations agent
    if not new_state.author_relations:
        author_relations_id = generate_id("author_relations")
        new_state.author_relations[author_relations_id] = AuthorRelationsAgentState(
            agent_id=author_relations_id, agent_role=AgentRole.AUTHOR_RELATIONS, status="idle"
        )

    # Style guide editor
    if not new_state.style_guide_editor:
        style_guide_id = generate_id("style_guide")
        new_state.style_guide_editor[style_guide_id] = StyleGuideEditorState(
            agent_id=style_guide_id, agent_role=AgentRole.STYLE_GUIDE_EDITOR, status="idle"
        )

    # Human in the loop
    if not new_state.human_in_loop:
        human_id = generate_id("human")
        new_state.human_in_loop[human_id] = HumanInLoopState(
            agent_id=human_id, agent_role=AgentRole.HUMAN_IN_LOOP, status="ready"
        )

    # Initialize story state based on operation mode
    if new_state.operation_mode == OperationMode.CREATE:
        # For new story creation
        new_state.current_state = StoryState.INITIATED
        if not new_state.story:
            title = new_state.user_request.title if new_state.user_request else "Untitled Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content="",
                state=StoryState.INITIATED,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    elif new_state.operation_mode == OperationMode.IMPORT:
        # For importing existing content
        new_state.current_state = StoryState.PLANNING
        if new_state.user_request and new_state.user_request.existing_content:
            new_state.imported_content = new_state.user_request.existing_content

            # Create a story object with the imported content
            title = new_state.user_request.title if new_state.user_request else "Imported Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content=new_state.imported_content,
                state=StoryState.PLANNING,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    elif new_state.operation_mode in [OperationMode.EDIT, OperationMode.CONTINUE]:
        # For editing or continuing an existing story
        new_state.current_state = StoryState.REVISION
        if new_state.user_request:
            if new_state.user_request.existing_content:
                new_state.imported_content = new_state.user_request.existing_content
            if new_state.user_request.sections_to_edit:
                new_state.sections_to_edit = new_state.user_request.sections_to_edit

            # Create a story object with the existing content
            title = new_state.user_request.title if new_state.user_request else "Edited Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content=new_state.imported_content or "",
                state=StoryState.REVISION,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    # Initialize empty bible if none exists
    if not new_state.bible:
        new_state.bible = StoryBible(
            story_id=new_state.story_id,
            created_at=current_timestamp(),
            updated_at=current_timestamp(),
        )

    return new_state

# Define the state graph
state_graph = StateGraph(initial_state=GraphState)

# Add nodes to the graph
state_graph.add_node("initialize_workflow", initialize_workflow)
state_graph.add_node("process_user_request", process_user_request)
state_graph.add_node("analyze_imported_content", analyze_imported_content)
state_graph.add_node("create_reverse_outline", create_reverse_outline)
state_graph.add_node("plan_edit_continuation", plan_edit_continuation)
state_graph.add_node("execute_edit_continuation", execute_edit_continuation)
state_graph.add_node("review_edit_continuation", review_edit_continuation)
state_graph.add_node("process_human_edit_approval", process_human_edit_approval)
state_graph.add_node("conduct_briefing", conduct_briefing)
state_graph.add_node("process_human_briefing_input", process_human_briefing_input)
state_graph.add_node("start_publishing_phase", start_publishing_phase)
state_graph.add_node("complete_project", complete_project)

# Define transitions between nodes
state_graph.add_transition(END, "initialize_workflow")
state_graph.add_transition("initialize_workflow", "process_user_request")
state_graph.add_transition("process_user_request", "conduct_briefing", condition=lambda state: state.operation_mode == OperationMode.CREATE)
state_graph.add_transition("conduct_briefing", "process_human_briefing_input")
state_graph.add_transition("process_human_briefing_input", "plan_edit_continuation", condition=lambda state: state.current_state == StoryState.PLANNING)
state_graph.add_transition("process_user_request", "analyze_imported_content", condition=lambda state: state.operation_mode == OperationMode.IMPORT)
state_graph.add_transition("analyze_imported_content", "create_reverse_outline")
state_graph.add_transition("create_reverse_outline", "plan_edit_continuation")
state_graph.add_transition("process_user_request", "plan_edit_continuation", condition=lambda state: state.operation_mode in [OperationMode.EDIT, OperationMode.CONTINUE])
state_graph.add_transition("plan_edit_continuation", "execute_edit_continuation")
state_graph.add_transition("execute_edit_continuation", "review_edit_continuation")
state_graph.add_transition("review_edit_continuation", "process_human_edit_approval", condition=lambda state: True)
state_graph.add_transition("process_human_edit_approval", "start_publishing_phase", condition=lambda state: state.current_state == StoryState.READY_FOR_PUBLISHING)
state_graph.add_transition("start_publishing_phase", "complete_project")
state_graph.add_transition("complete_project", END)

# Save the state graph to a file for persistence
graph_file_path = "output/story_graph_state.json"
try:
    os.makedirs(os.path.dirname(graph_file_path), exist_ok=True)
    with open(graph_file_path, "w", encoding="utf-8") as file:
        json.dump(state_graph.to_dict(), file)
except Exception as e:
    error_msg = f"Error saving state graph: {str(e)}"
    print(error_msg)

# Function to load state graph from a file
def load_state_graph(file_path: str) -> StateGraph:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            graph_data = json.load(file)
            return StateGraph.from_dict(graph_data)
    except Exception as e:
        error_msg = f"Error loading state graph: {str(e)}"
        print(error_msg)
        return StateGraph(initial_state=GraphState)

# Load the state graph if it exists
if os.path.exists(graph_file_path):
    state_graph = load_state_graph(graph_file_path)

# Main execution function
def execute_story_workflow(user_request: UserRequest):
    # Initialize workflow state
    state = GraphState()
    state = initialize_workflow(state)

    # Process the user request
    state = process_user_request(state, user_request)

    # Execute the state graph
    current_node = state_graph.initial_node
    while current_node != END:
        state = state_graph.execute_node(current_node, state)
        next_node = state_graph.get_next_node(current_node, state)
        current_node = next_node

    # Save the final state graph
    try:
        with open(graph_file_path, "w", encoding="utf-8") as file:
            json.dump(state_graph.to_dict(), file)
    except Exception as e:
        error_msg = f"Error saving final state graph: {str(e)}"
        print(error_msg)

    return state

# Example user request
example_request = UserRequest(
    user_id="user_123",
    title="The Adventure of the Brave Knight",
    operation_mode=OperationMode.CREATE,
    story_structure=StoryStructure.THREE_ACT,
    num_writers=2,
    use_joint_llm=False,
)

# Execute the workflow with the example request
final_state = execute_story_workflow(example_request)
print(f"Final story state: {final_state.story.state}")
print(f"Final story content:\n{final_state.story.content[:500]}...")

# Interactive loop for user queries
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Stream graph updates based on user input
        events = state_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
                
    except Exception as e:
        print(f"Error: {str(e)}")
        break
