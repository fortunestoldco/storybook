from typing import Dict, List, Any
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

llm = get_llm()

# ----------------------------------------------------------------------
# 1. Define Node Functions (Workflow Steps)
# ----------------------------------------------------------------------

def author_relations_brainstorm(state: StoryCreationState) -> Dict:
    """Simulates a brainstorming session with the Author Relations Officer."""
    prompt = f"""You are the Author Relations Officer. The user has provided the following story brief:
    Title: {state.title}
    Genre: {state.genre}
    Themes: {', '.join(state.themes)}

    Brainstorm with the Overall Supervisor (simulated) to generate additional ideas, overall goals, and extra thoughts for the story.
    Provide a concise summary of the brainstorming session.
    """
    brainstorm_summary = llm.invoke(prompt).content
    add_to_story_bible(brainstorm_summary, {"step": "author_relations_brainstorm"})
    return {"brainstorm_summary": brainstorm_summary}

def project_initiation(state: StoryCreationState) -> Dict:
    """Initializes the story creation project by creating a Story Bible and setting initial configurations."""
    story_bible = f"Story Title: {state.title}\nGenre: {state.genre}\nThemes: {', '.join(state.themes)}\nBrainstorm Summary: {state.brainstorm_summary}"
    add_to_story_bible(story_bible, {"step": "project_initiation"})
    return {
        "story_bible": story_bible,
        "current_step": "market_research",
        "project_status": "In Research",
        "project_progress": 0,
    }

def market_research(state: StoryCreationState) -> Dict:
    """Performs market research to understand industry trends and commercial opportunities."""
    prompt = MARKET_RESEARCH_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "market_research"})
    return {
        "market_research_report": report,
        "current_step": "contextual_research",
        "project_progress": 10,
    }

def contextual_research(state: StoryCreationState) -> Dict:
    """Researches similar books and analyzes their commercial and critical results."""
    prompt = CONTEXTUAL_RESEARCH_PROMPT
    report = llm.invoke(prompt).content
    example_url = "https://www.example.com"
    web_content = web_crawl(example_url)
    report += f"\nExample crawled content from {example_url}: {web_content[:200]}..."
    add_to_story_bible(report, {"step": "contextual_research"})
    return {
        "contextual_research_report": report,
        "current_step": "consumer_research",
        "project_progress": 20,
    }

def consumer_research(state: StoryCreationState) -> Dict:
    """Determines the target market and provides contextual information about them."""
    prompt = CONSUMER_RESEARCH_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "consumer_research"})
    return {
        "consumer_research_report": report,
        "current_step": "research_supervisor_review",
        "project_progress": 30,
    }

def research_supervisor_review(state: StoryCreationState) -> Dict:
    """Research Supervisor compiles the research reports."""
    compiled_report = f"""Market Research Report:\n{state.market_research_report}
    \nContextual Research Report:\n{state.contextual_research_report}
    \nConsumer Research Report:\n{state.consumer_research_report}"""
    add_to_story_bible(compiled_report, {"step": "research_supervisor_review"})
    return {
        "research_reports": [compiled_report],
        "current_step": "overall_supervisor_review",
        "project_progress": 40,
    }

def overall_supervisor_review(state: StoryCreationState) -> Dict:
    """The Overall Supervisor updates the project status and gives the story bible to the Thematics and Narrative Team."""
    return {
        "current_step": "world_building",
        "project_status": "World Building",
        "project_progress": 42,
        "story_bible": state.story_bible,
    }

def world_building(state: StoryCreationState) -> Dict:
    """Thematics and Narrative team - World Building Writer."""
    prompt = WORLD_BUILDING_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "world_building"})
    return {
        "world_specification": report,
        "current_step": "character_development",
        "project_progress": 44,
    }

def character_development(state: StoryCreationState) -> Dict:
    """Thematics and Narrative team - Character Development Writer."""
    prompt = CHARACTER_DEVELOPMENT_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "character_development"})
    return {
        "character_development": report,
        "current_step": "writing_team_supervisor_review",
        "project_progress": 46,
    }

def writing_team_supervisor_review(state: StoryCreationState) -> Dict:
    """Writing Team Supervisor defines the overall story plan."""
    story_plan = f"""World Specification:\n{state.world_specification}\n\nCharacter Development:\n{state.character_development}"""
    add_to_story_bible(story_plan, {"step": "writing_team_supervisor_review"})
    return {
        "story_plan": story_plan,
        "current_step": "editorial_review_initial",
        "project_progress": 48,
    }

def editorial_review_initial(state: StoryCreationState) -> Dict:
    """Editorial Team reviews the story plan, world specification, and character development."""
    prompt = EDITORIAL_FEEDBACK_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "editorial_review_initial"})
    editorial_grade = "good"  # Replace with actual grading logic
    return {
        "editorial_grade": editorial_grade,
        "editorial_notes_initial": report,
        "current_step": "handle_editorial_grade",
        "project_progress": 50,
    }

def handle_editorial_grade(state: StoryCreationState) -> Dict:
    """Handles the branching logic for editorial grades."""
    grade = state.editorial_grade
    if grade == "low":
        next_step = "world_building"
    elif grade == "middling":
        next_step = "writing_team_supervisor_review"
    else:
        next_step = "author_liaison_chat_session"
    return {"current_step": next_step}

def author_liaison_chat_session(state: StoryCreationState) -> Dict:
    """Author liaison chat session with user."""
    return {
        "current_step": "story_and_dialogue_writers_draft_chapters",
        "project_status": "Manuscript Writing",
        "project_progress": 52,
    }

def story_and_dialogue_writers_draft_chapters(state: StoryCreationState) -> Dict:
    """Story writers and dialogue writers start on chapters."""
    chapter_1_number = state.chapters_written + 1
    chapter_2_number = state.chapters_written + 2

    # Chapter 1
    prompt_story = STORY_WRITER_PROMPT.format(chapter_number=chapter_1_number)
    chapter_1_story = llm.invoke(prompt_story).content
    prompt_dialogue = DIALOGUE_WRITER_PROMPT.format(chapter_number=chapter_1_number)
    chapter_1_dialogue = llm.invoke(prompt_dialogue).content

    # Chapter 2
    prompt_story_2 = STORY_WRITER_PROMPT.format(chapter_number=chapter_2_number)
    chapter_2_story = llm.invoke(prompt_story_2).content
    prompt_dialogue_2 = DIALOGUE_WRITER_PROMPT.format(chapter_number=chapter_2_number)
    chapter_2_dialogue = llm.invoke(prompt_dialogue_2).content

    chapter_1_draft = f"""Story Writer: {chapter_1_story}\n\nDialogue Writer: {chapter_1_dialogue}"""
    chapter_2_draft = f"""Story Writer: {chapter_2_story}\n\nDialogue Writer: {chapter_2_dialogue}"""

    add_to_story_bible(chapter_1_draft, {"step": "story_and_dialogue_writers_draft_chapters", "chapter": chapter_1_number})
    add_to_story_bible(chapter_2_draft, {"step": "story_and_dialogue_writers_draft_chapters", "chapter": chapter_2_number})

    chapter_drafts = state.chapter_drafts.copy()
    chapter_drafts[chapter_1_number] = chapter_1_draft
    chapter_drafts[chapter_2_number] = chapter_2_draft

    return {
        "chapter_drafts": chapter_drafts,
        "current_step": "editorial_team_supervisor_review_chapters",
        "chapters_written": chapter_2_number,
        "project_progress": 54,
    }

def editorial_team_supervisor_review_chapters(state: StoryCreationState) -> Dict:
    """Editorial team supervisor review chapters."""
    chapter_1_number = state.chapters_written - 1
    chapter_2_number = state.chapters_written

    prompt_continuity = CONTINUITY_CHECKER_PROMPT.format(
        chapter_1_number=chapter_1_number, chapter_2_number=chapter_2_number
    )
    report_continuity = llm.invoke(prompt_continuity).content

    prompt_cohesiveness = COHESIVENESS_CHECKER_PROMPT.format(
        chapter_1_number=chapter_1_number, chapter_2_number=chapter_2_number
    )
    report_cohesiveness = llm.invoke(prompt_cohesiveness).content

    prompt_editorial_feedback = CHAPTER_EDITORIAL_FEEDBACK_PROMPT.format(
        chapter_1_number=chapter_1_number, chapter_2_number=chapter_2_number
    )
    report_editorial_feedback = llm.invoke(prompt_editorial_feedback).content

    editorial_notes_chapter = state.editorial_notes.copy()

    if state.overall_manuscript_draft_number < 3 and state.chapters_written < 10:
        # Process chapter 1 notes
        if chapter_1_number not in editorial_notes_chapter:
            editorial_notes_chapter[chapter_1_number] = []
        editorial_notes_chapter[chapter_1_number].extend([
            report_continuity,
            report_cohesiveness,
            report_editorial_feedback
        ])

        # Process chapter 2 notes
        if chapter_2_number not in editorial_notes_chapter:
            editorial_notes_chapter[chapter_2_number] = []
        editorial_notes_chapter[chapter_2_number].extend([
            report_continuity,
            report_cohesiveness,
            report_editorial_feedback
        ])

        return {
            "editorial_notes": editorial_notes_chapter,
            "current_step": "story_and_dialogue_writers_draft_chapters",
            "project_progress": 56,
        }
    else:
        return {
            "current_step": "complete",
            "project_progress": 100,
            "all_chapters_finished": True,
        }

def writing_phase(state: StoryCreationState) -> Dict:
    """Writing phase of the story creation process."""
    return {
        "current_step": "writing_phase",
        "project_status": "Writing in Progress",
        "project_progress": 60
    }

def editing_phase(state: StoryCreationState) -> Dict:
    """Editing phase of the story creation process."""
    return {
        "current_step": "editing_phase",
        "project_status": "Editing in Progress",
        "project_progress": 75
    }

def final_review_phase(state: StoryCreationState) -> Dict:
    """Final review phase of the story creation process."""
    return {
        "current_step": "final_review_phase",
        "project_status": "Final Review",
        "project_progress": 90
    }

def publishing_phase(state: StoryCreationState) -> Dict:
    """Publishing phase of the story creation process."""
    return {
        "current_step": "publishing_phase",
        "project_status": "Publishing",
        "project_progress": 95
    }

def human_validation(state: StoryCreationState) -> Dict:
    """Pauses the workflow for human validation."""
    return {
        "human_validation": True,
        "current_step": f"Awaiting Human Validation: {state.current_step}",
    }

def update_story_bible_node(state: StoryCreationState) -> Dict:
    """Update the story bible with the new information."""
    updated_story_bible = update_story_bible(
        story_bible=state.story_bible, new_information=state.input
    )
    add_to_story_bible(state.input, {"step": "update_story_bible"})
    return {
        "story_bible": updated_story_bible,
        "human_validation": False,
        "input": "",
    }

def complete(state: StoryCreationState) -> Dict:
    """Completes the story and does final touches."""
    final_manuscript = str(state.chapter_drafts)
    return {
        "final_manuscript": final_manuscript,
        "current_step": "complete",
        "project_status": "Complete",
        "project_progress": 100,
    }

# ----------------------------------------------------------------------
# 2. Define Edge Functions
# ----------------------------------------------------------------------

def should_validate(state: StoryCreationState) -> str:
    """Determines if human validation is needed."""
    return "human_validation" if state.human_validation else "continue"

def has_input(state: StoryCreationState) -> str:
    """Determines if there's new user input to update the story bible."""
    return "yes" if state.input else "no"

def handle_editorial_grade_logic(state: StoryCreationState) -> str:
    """Handles the branching logic for editorial grades."""
    if state.editorial_grade == "low":
        return "world_building"
    elif state.editorial_grade == "middling":
        return "writing_team_supervisor_review"
    return "author_liaison_chat_session"

def check_all_chapters_written(state: StoryCreationState) -> str:
    """Are all chapters finished?"""
    return "complete" if state.all_chapters_finished else "story_and_dialogue_writers_draft_chapters"

# ----------------------------------------------------------------------
# 3. Create and Configure the Graph
# ----------------------------------------------------------------------

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

# Add all nodes to the builder
for node_name, node_function in nodes.items():
    builder.add_node(node_name, node_function)

# Add main sequence edges
main_sequence = [
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
    ("author_liaison_chat_session", "story_and_dialogue_writers_draft_chapters"),
    ("story_and_dialogue_writers_draft_chapters", "editorial_team_supervisor_review_chapters"),
    ("complete", "human_validation")
]

# Add the main sequence edges
for start, end in main_sequence:
    builder.add_edge(start, end)

# Add conditional edges for editorial grade handling
builder.add_conditional_edges(
    "handle_editorial_grade",
    handle_editorial_grade_logic,
    {
        "world_building": "world_building",
        "writing_team_supervisor_review": "writing_team_supervisor_review",
        "author_liaison_chat_session": "author_liaison_chat_session"
    }
)

# Add conditional edges for chapter review cycle
builder.add_conditional_edges(
    "editorial_team_supervisor_review_chapters",
    check_all_chapters_written,
    {
        "story_and_dialogue_writers_draft_chapters": "story_and_dialogue_writers_draft_chapters",
        "complete": "complete"
    }
)

# Add human validation conditional edges
builder.add_conditional_edges(
    "human_validation",
    should_validate,
    {
        "continue": "update_story_bible",
        "human_validation": "human_validation"
    }
)

# Add story bible update conditional edges
builder.add_conditional_edges(
    "update_story_bible",
    has_input,
    {
        "yes": "update_story_bible",
        "no": {
            "market_research": "contextual_research",
            "contextual_research": "consumer_research",
            "consumer_research": "research_supervisor_review",
            "research_supervisor_review": "overall_supervisor_review",
            "overall_supervisor_review": "world_building",
            "world_building": "character_development",
            "character_development": "writing_team_supervisor_review",
            "writing_team_supervisor_review": "editorial_review_initial",
            "editorial_review_initial": "handle_editorial_grade",
            "handle_editorial_grade": "world_building",
            "author_liaison_chat_session": "story_and_dialogue_writers_draft_chapters",
            "story_and_dialogue_writers_draft_chapters": "editorial_team_supervisor_review_chapters",
            "writing_phase": "editing_phase",
            "editing_phase": "final_review_phase",
            "final_review_phase": "publishing_phase",
            "complete": "human_validation"
        }[state.current_step]
    }
)

# Add validation phase edges
validation_edges = [
    ("writing_phase", "human_validation"),
    ("editing_phase", "human_validation"),
    ("final_review_phase", "human_validation"),
]

for start, end in validation_edges:
    builder.add_edge(start, end)

# ----------------------------------------------------------------------
# 4. Graph Execution Functions
# ----------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Builds and returns the configured StateGraph.
    
    Returns:
        StateGraph: The fully configured graph ready for execution
    """
    try:
        return builder.compile()
    except Exception as e:
        print(f"Error building graph: {str(e)}")
        raise

def run_graph(graph: StateGraph, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Runs the graph with the provided initial state.
    
    Args:
        graph (StateGraph): The compiled state graph to run
        initial_state (Dict[str, Any], optional): Initial state to start the graph with.
                                               Defaults to None.
    
    Returns:
        Dict[str, Any]: The final state after running the graph
    """
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

# Create and export the graph instance
graph = build_graph()

if __name__ == "__main__":
    # Initialize the vectorstore
    get_story_bible_vectorstore()

    # Create initial state for testing
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

    # Run the graph with test state
    try:
        result = run_graph(graph, test_state)
        print(f"Current Step: {result['current_step']}")
        print(f"Project Status: {result['project_status']}")
        print(f"Project Progress: {result['project_progress']}%")
    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
