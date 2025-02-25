from typing import Dict, List, Any, TypedDict, Literal, Optional, Union
from langgraph.graph import StateGraph, MessageGraph
from storybook.config import Config
from storybook.state import StoryCreationState
from storybook.prompts import (
    RESEARCH_PROMPT,
    OUTLINE_PROMPT,
    WRITING_PROMPT,
    EDITING_PROMPT,
    MARKET_RESEARCH_PROMPT,
    CONTEXTUAL_RESEARCH_PROMPT,
    CONSUMER_RESEARCH_PROMPT,
    WORLD_BUILDING_PROMPT,
    CHARACTER_DEVELOPMENT_PROMPT,
    STORY_WRITER_PROMPT,
    DIALOGUE_WRITER_PROMPT,
    CONTINUITY_CHECKER_PROMPT,
    COHESIVENESS_CHECKER_PROMPT,
    EDITORIAL_FEEDBACK_PROMPT,
    CHAPTER_EDITORIAL_FEEDBACK_PROMPT,
)
from storybook.utils import (
    get_llm,
    consolidate_sections,
    add_to_story_bible,
    get_story_bible_vectorstore,
    web_crawl,
)
from storybook.tools import update_story_bible
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool

class ActionRequest(TypedDict):
    action: str
    args: dict

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]

class HumanResponse(TypedDict):
    type: Literal['accept', 'ignore', 'response', 'edit']
    args: Union[None, str, ActionRequest]

llm = get_llm()

def human_validation(state: StoryCreationState) -> Dict:
    """Pauses the workflow for human validation using Agent Inbox interrupts."""
    current_step = state.current_step
    
    request: HumanInterrupt = {
        "action_request": {
            "action": f"Review {current_step}",
            "args": {
                "step": current_step,
                "content": state.story_bible,
                "progress": state.project_progress,
                "status": state.project_status
            }
        },
        "config": {
            "allow_ignore": False,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True
        },
        "description": f"{current_step} Review Required\n\nCurrent project status: {state.project_status}\nProgress: {state.project_progress}%\n\nPlease review the current state of the story and provide your feedback:\n\n{state.story_bible}\n\nActions Available:\n- Accept: Approve the current state and continue\n- Edit: Make changes to the content\n- Respond: Provide feedback or instructions"
    }

    response = interrupt([request])[0]

    if response["type"] == "accept":
        return {
            "human_validation": False,
            "current_step": state.current_step
        }
    elif response["type"] == "edit":
        edit_request = response["args"]
        if isinstance(edit_request, dict):
            return {
                "input": edit_request.get("content", ""),
                "human_validation": False,
                "current_step": state.current_step
            }
    elif response["type"] == "response":
        feedback = response["args"]
        if isinstance(feedback, str):
            add_to_story_bible(feedback, {"step": "human_feedback"})
        return {
            "input": feedback if isinstance(feedback, str) else "",
            "human_validation": False,
            "current_step": state.current_step
        }
    
    return {
        "human_validation": True,
        "current_step": f"Awaiting Human Validation: {state.current_step}"
    }

def author_liaison_chat_session(state: StoryCreationState) -> Dict:
    """Author liaison chat session with user using Agent Inbox interrupts."""
    request: HumanInterrupt = {
        "action_request": {
            "action": "Author Liaison Session",
            "args": {
                "story_bible": state.story_bible,
                "editorial_grade": state.editorial_grade
            }
        },
        "config": {
            "allow_ignore": False,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True
        },
        "description": f"Author Liaison Session\n\nThe editorial team has reviewed your story with a grade of: {state.editorial_grade}\n\nCurrent Story Bible:\n{state.story_bible}\n\nPlease provide your input on the story direction:\n- Review the current story bible\n- Suggest any changes or additions\n- Provide guidance for the next phase"
    }

    response = interrupt([request])[0]

    if response["type"] == "accept":
        return {
            "current_step": "story_and_dialogue_writers_draft_chapters",
            "project_status": "Manuscript Writing",
            "project_progress": 52
        }
    elif response["type"] in ["edit", "response"]:
        feedback = response["args"]
        if isinstance(feedback, (str, dict)):
            content = feedback if isinstance(feedback, str) else str(feedback)
            add_to_story_bible(content, {"step": "author_liaison_feedback"})
        
    return {
        "current_step": "story_and_dialogue_writers_draft_chapters",
        "project_status": "Manuscript Writing",
        "project_progress": 52,
    }

builder = StateGraph(StoryCreationState)

nodes = {
    "author_relations_brainstorm": author_relations_brainstorm,
    "project_initiation": project_initiation,
    "market_research": market_research,
    "contextual_research": contextual_research,
    "consumer_research": consumer_research,
    "research_supervisor_review": research_supervisor_review,
    "overall_supervisor_review": overall_supervisor_review,
    "world_building": world_building,
    "character_development": character_development,
    "writing_team_supervisor_review": writing_team_supervisor_review,
    "editorial_review_initial": editorial_review_initial,
    "handle_editorial_grade": handle_editorial_grade,
    "author_liaison_chat_session": author_liaison_chat_session,
    "story_and_dialogue_writers_draft_chapters": story_and_dialogue_writers_draft_chapters,
    "editorial_team_supervisor_review_chapters": editorial_team_supervisor_review_chapters,
    "writing_phase": writing_phase,
    "editing_phase": editing_phase,
    "final_review_phase": final_review_phase,
    "publishing_phase": publishing_phase,
    "human_validation": human_validation,
    "update_story_bible": update_story_bible_node,
    "complete": complete
}

for node_name, node_function in nodes.items():
    builder.add_node(node_name, node_function)

edges = [
    ("author_relations_brainstorm", "project_initiation"),
    ("project_initiation", "market_research"),
    ("market_research", "contextual_research"),
    ("contextual_research", "consumer_research"),
    ("consumer_research", "research_supervisor_review"),
    ("research_supervisor_review", "overall_supervisor_review"),
    ("overall_supervisor_review", "world_building"),
    ("world_building", "character_development"),
    ("character_development", "writing_team_supervisor_review"),
    ("writing_team_supervisor_review", "editorial_review_initial"),
    ("editorial_review_initial", "handle_editorial_grade"),
]

for start, end in edges:
    builder.add_edge(start, end)

builder.add_conditional_edges(
    "handle_editorial_grade",
    handle_editorial_grade_logic,
    {
        "world_building": "world_building",
        "writing_team_supervisor_review": "writing_team_supervisor_review",
        "author_liaison_chat_session": "author_liaison_chat_session"
    }
)

builder.add_conditional_edges(
    "human_validation",
    should_validate,
    {
        "continue": "update_story_bible",
        "human_validation": "human_validation"
    }
)

def build_graph() -> StateGraph:
    try:
        return builder.compile()
    except Exception as e:
        print(f"Error building graph: {str(e)}")
        raise

def run_graph(graph: StateGraph, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        if initial_state is None:
            initial_state = {
                "title": "",
                "genre": "",
                "themes": [],
                "current_step": "author_relations_brainstorm",
                "project_status": "Not Started",
                "project_progress": 0,
                "chapters_written": 0,
                "chapter_drafts": {},
                "editorial_notes": {},
                "overall_manuscript_draft_number": 0,
                "all_chapters_finished": False,
                "human_validation": False,
                "input": "",
            }

        config = RunnableConfig(
            recursion_limit=100,
            tags=["storybook_workflow"]
        )

        workflow = graph.configurable_chain()
        result = workflow.invoke(initial_state, config=config)
        
        return result
    except Exception as e:
        print(f"Error running graph: {str(e)}")
        raise

graph = build_graph()

if __name__ == "__main__":
    get_story_bible_vectorstore()

    test_state = {
        "title": "The AI Detective",
        "genre": "Science Fiction Mystery",
        "themes": ["Artificial Intelligence", "Moral Dilemmas", "Future Crimes"],
        "current_step": "author_relations_brainstorm",
        "project_status": "Not Started",
        "project_progress": 0,
        "chapters_written": 0,
        "chapter_drafts": {},
        "editorial_notes": {},
        "overall_manuscript_draft_number": 0,
        "all_chapters_finished": False,
        "human_validation": False,
        "input": "",
    }

    try:
        result = run_graph(graph, test_state)
        print(f"Current Step: {result['current_step']}")
        print(f"Project Status: {result['project_status']}")
        print(f"Project Progress: {result['project_progress']}%")
    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
