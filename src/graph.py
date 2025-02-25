from typing import Dict, List
from langgraph.graph import StateGraph, MessageGraph
from storybook.state import StoryCreationState
from storybook.prompts import RESEARCH_PROMPT, OUTLINE_PROMPT, WRITING_PROMPT, EDITING_PROMPT, MARKET_RESEARCH_PROMPT, CONTEXTUAL_RESEARCH_PROMPT, CONSUMER_RESEARCH_PROMPT, WORLD_BUILDING_PROMPT, CHARACTER_DEVELOPMENT_PROMPT, STORY_WRITER_PROMPT, DIALOGUE_WRITER_PROMPT, CONTINUITY_CHECKER_PROMPT, COHESIVENESS_CHECKER_PROMPT, EDITORIAL_FEEDBACK_PROMPT, CHAPTER_EDITORIAL_FEEDBACK_PROMPT
from storybook.utils import get_llm, consolidate_sections, add_to_story_bible, get_story_bible_vectorstore, web_crawl
from storybook.tools import update_story_bible
from langchain_core.runnables import chain


llm = get_llm()

# ----------------------------------------------------------------------
# 3. Define Nodes (Workflow Steps) - Reorganized for clarity
# ----------------------------------------------------------------------

def author_relations_brainstorm(state: StoryCreationState) -> Dict:
    """Simulates a brainstorming session with the Author Relations Officer."""
    # Here, you'd integrate the actual communication with a human (e.g., via a chatbot interface).
    # For this example, we'll simulate it with an LLM.
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
    story_bible = f"Story Title: {state.title}\nGenre: {state.genre}\nThemes: {', '.join(state.themes)}\nBrainstorm Summary: {state.brainstorm_summary}" #Incorporates brainstorming summary
    add_to_story_bible(story_bible, {"step": "project_initiation"})  # Store in MongoDB
    return {"story_bible": story_bible, "current_step": "Market Research", "project_status": "In Research", "project_progress": 0, "project_goal": "Create a best-selling class novel that ticks all the boxes."} # added project goal

def market_research(state: StoryCreationState) -> Dict:
    """Performs market research to understand industry trends and commercial opportunities."""
    prompt = MARKET_RESEARCH_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "market_research"}) # Store in MongoDB
    return {"market_research_report": report, "current_step": "Contextual Research", "project_progress": 10} #Set to 10% as example

def contextual_research(state: StoryCreationState) -> Dict:
    """Researches similar books and analyzes their commercial and critical results."""
    prompt = CONTEXTUAL_RESEARCH_PROMPT
    report = llm.invoke(prompt).content

    #Web crawl and use document storage agent
    example_url = "https://www.example.com" #Replace with your website
    web_content = web_crawl(example_url)

    report += f"\nExample crawled content from {example_url}: {web_content[:200]}..."

    add_to_story_bible(report, {"step": "contextual_research"})
    return {"contextual_research_report": report, "current_step": "Consumer Research", "project_progress": 20}

def consumer_research(state: StoryCreationState) -> Dict:
    """Determines the target market and provides contextual information about them."""
    prompt = CONSUMER_RESEARCH_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "consumer_research"})
    return {"consumer_research_report": report, "current_step": "Research Supervisor Review", "project_progress": 30}

def research_supervisor_review(state: StoryCreationState) -> Dict:
    """Research Supervisor compiles the research reports."""
    compiled_report = f"""Market Research Report:\n{state.market_research_report}\n\nContextual Research Report:\n{state.contextual_research_report}\n\nConsumer Research Report:\n{state.consumer_research_report}"""
    add_to_story_bible(compiled_report, {"step": "research_supervisor_review"})
    return {"research_reports": [compiled_report], "current_step": "Overall Supervisor Review", "project_progress": 40} #Store compiled research report into the research reports array.


def overall_supervisor_review(state: StoryCreationState) -> Dict:
    """The Overall Supervisor updates the project status and gives the story bible to the Thematics and Narrative Team."""
    project_status = "World Building"
    project_progress = 42 # set a little bit above the last progress
    return {"current_step": "World Building", "project_status": project_status, "project_progress": project_progress, "story_bible": state.story_bible}

def world_building(state: StoryCreationState) -> Dict:
    """Thematics and Narrative team - World Building Writer."""
    prompt = WORLD_BUILDING_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "world_building"})
    return {"world_specification": report, "current_step": "Character Development", "project_progress": 44}

def character_development(state: StoryCreationState) -> Dict:
    """Thematics and Narrative team - Character Development Writer."""
    prompt = CHARACTER_DEVELOPMENT_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "character_development"})
    return {"character_development": report, "current_step": "Writing Team Supervisor Review", "project_progress": 46}

def writing_team_supervisor_review(state: StoryCreationState) -> Dict:
  """Writing Team Supervisor defines the overall story plan."""
  story_plan = f"""World Specification:\n{state.world_specification}\n\nCharacter Development:\n{state.character_development}"""
  add_to_story_bible(story_plan, {"step": "writing_team_supervisor_review"})
  return {"story_plan": story_plan, "current_step": "Editorial Review (Initial)", "project_progress": 48}

def editorial_review_initial(state: StoryCreationState) -> Dict:
    """Editorial Team reviews the story plan, world specification, and character development."""
    prompt = EDITORIAL_FEEDBACK_PROMPT
    report = llm.invoke(prompt).content
    add_to_story_bible(report, {"step": "editorial_review_initial"})

    #Simulate Editorial Grade (low, middling, good, amazing)
    editorial_grade = "good" # Replace with actual grading logic
    return {"editorial_grade": editorial_grade, "editorial_notes_initial": report, "current_step": f"Handle Editorial Grade ({editorial_grade})", "project_progress": 50}

def handle_editorial_grade(state: StoryCreationState) -> Dict:
  """Branching depending on the editorial grade:"""
  grade = state.editorial_grade

  if grade == "low":
    next_step = "World Building"
  elif grade == "middling":
    next_step = "Writing Team Supervisor Review" #Rewrite = Writing Team Supervisor Review
  else:
    next_step = "Author Liaison Chat Session" # Good or amazing =  Author Liason chat
  return {"current_step": next_step}

def author_liaison_chat_session(state: StoryCreationState) -> Dict:
  """Author liason chat session with user"""

  project_status = "Manuscript Writing" # Move to writing phase
  project_progress = 52
  return {"current_step": "Story and Dialogue Writers Draft Chapters", "project_status": project_status, "project_progress": project_progress}

def story_and_dialogue_writers_draft_chapters(state: StoryCreationState) -> Dict:
    """Story writers and dialogue writers start on the first two chapters."""

    chapter_1_number = state.chapters_written + 1 #Chapter 1 #This should be a global incremental change
    chapter_2_number = state.chapters_written + 2 #Chapter 2

    prompt_story = STORY_WRITER_PROMPT.format(chapter_number = chapter_1_number)
    chapter_1_story = llm.invoke(prompt_story).content # Get story writers opinion.
    prompt_dialogue = DIALOGUE_WRITER_PROMPT.format(chapter_number = chapter_1_number)
    chapter_1_dialogue = llm.invoke(prompt_dialogue).content # Get dialogue writers opinion.

    prompt_story_2 = STORY_WRITER_PROMPT.format(chapter_number = chapter_2_number)
    chapter_2_story = llm.invoke(prompt_story_2).content
    prompt_dialogue_2 = DIALOGUE_WRITER_PROMPT.format(chapter_number = chapter_2_number)
    chapter_2_dialogue = llm.invoke(prompt_dialogue_2).content


    chapter_1_draft = f"""Story Writer: {chapter_1_story}\n\nDialogue Writer: {chapter_1_dialogue}"""
    chapter_2_draft = f"""Story Writer: {chapter_2_story}\n\nDialogue Writer: {chapter_2_dialogue}"""

    add_to_story_bible(chapter_1_draft, {"step": "story_and_dialogue_writers_draft_chapters", "chapter": chapter_1_number})
    add_to_story_bible(chapter_2_draft, {"step": "story_and_dialogue_writers_draft_chapters", "chapter": chapter_2_number}) # Stores the chapter draft to the book bible

    chapter_drafts = state.chapter_drafts.copy()
    chapter_drafts[chapter_1_number] = chapter_1_draft
    chapter_drafts[chapter_2_number] = chapter_2_draft

    return {"chapter_drafts": chapter_drafts, "current_step": "Editorial Team Supervisor Review (Chapters)", "chapters_written": chapter_2_number, "project_progress": 54}

def editorial_team_supervisor_review_chapters(state: StoryCreationState) -> Dict:
  """Editorial team supervisor review chapters
  """
  chapter_1_number = state.chapters_written - 1
  chapter_2_number = state.chapters_written

  prompt_continuity = CONTINUITY_CHECKER_PROMPT.format(chapter_1_number = chapter_1_number, chapter_2_number = chapter_2_number)
  report_continuity = llm.invoke(prompt_continuity).content # continuity report
  prompt_cohesiveness = COHESIVENESS_CHECKER_PROMPT.format(chapter_1_number = chapter_1_number, chapter_2_number = chapter_2_number)
  report_cohesiveness = llm.invoke(prompt_cohesiveness).content #cohesiveness report

  prompt_editorial_feedback = CHAPTER_EDITORIAL_FEEDBACK_PROMPT.format(chapter_1_number = chapter_1_number, chapter_2_number = chapter_2_number)
  report_editorial_feedback = llm.invoke(prompt_editorial_feedback).content #Chapter editorial feedback.
  editorial_grade = "good"

  editorial_notes_chapter = state.editorial_notes.copy() # Get the edit notes

  if state.overall_manuscript_draft_number < 3 and state.chapters_written <10: #Limit to 3 drafts, and only up to 10 chapter increments.
      if chapter_1_number not in editorial_notes_chapter:
          editorial_notes_chapter[chapter_1_number] = []
      editorial_notes_chapter[chapter_1_number].append(report_continuity)
      editorial_notes_chapter[chapter_1_number].append(report_cohesiveness)
      editorial_notes_chapter[chapter_1_number].append(report_editorial_feedback)

      if chapter_2_number not in editorial_notes_chapter:
        editorial_notes_chapter[chapter_2_number] = []
      editorial_notes_chapter[chapter_2_number].append(report_continuity)
      editorial_notes_chapter[chapter_2_number].append(report_cohesiveness)
      editorial_notes_chapter[chapter_2_number].append(report_editorial_feedback) # Append all notes.

      # Append all notes
      return {"editorial_notes": editorial_notes_chapter, "current_step": "Story and Dialogue Writers Draft Chapters", "project_progress": 56} # loops
  else:
      return {"current_step": "Complete", "project_progress": 100, "all_chapters_finished": True} # Stop!


def human_validation(state: StoryCreationState) -> Dict:
    """Pauses the workflow for human validation."""
    return {"human_validation": True, "current_step": f"Awaiting Human Validation: {state.current_step}"}

def update_story_bible_node(state: StoryCreationState) -> Dict:
    """Update the story bible with the new information."""
    updated_story_bible = update_story_bible(story_bible=state.story_bible, new_information=state.input)
    add_to_story_bible(state.input, {"step": "update_story_bible"})  # Store in MongoDB
    return {"story_bible": updated_story_bible, "human_validation": False, "input": ""} # Clear input.

def complete(state: StoryCreationState) -> Dict:
  """Completes the story and does final touches"""
  final_manuscript = str(state.chapter_drafts)
  return {"final_manuscript": final_manuscript, "current_step": "Final Touch up", "project_status": "Complete", "project_progress": 100}

# ----------------------------------------------------------------------
# 4. Define Edges (Transitions between Nodes)
# ----------------------------------------------------------------------

def should_validate(state: StoryCreationState) -> str:
    """Determines if human validation is needed."""
    if state.human_validation:
        return "human_validation"
    else:
        return "continue"

def has_input(state: StoryCreationState) -> str:
    """Determines if there's new user input to update the story bible."""
    if state.input:
        return "yes"
    else:
        return "no"

def handle_editorial_grade_logic(state: StoryCreationState) -> str:
    """Handles the branching logic for editorial grades"""
    editorial_grade = state.editorial_grade

    if editorial_grade == "low":
        return "world_building"
    elif editorial_grade == "middling":
        return "writing_team_supervisor_review"
    else: # good/amazing
        return "author_liaison_chat_session"

def check_all_chapters_written(state: StoryCreationState) -> str:
    """Are all chapters finished?"""
    if state.all_chapters_finished:
        return "complete"
    else:
        return "editorial_team_supervisor_review_chapters" #Go back to the drafts

# ----------------------------------------------------------------------
# 5. Create the LangGraph
# ----------------------------------------------------------------------

builder = StateGraph(StoryCreationState)

# Add Nodes

# Thematics and Narrative subgraph / writing team supervisor review (pre greenlight)
builder.add_node("world_building", world_building)
builder.add_node("character_development", character_development)
builder.add_node("writing_team_supervisor_review", writing_team_supervisor_review)

builder.add_node("editorial_review_initial", editorial_review_initial) # Editorial team
builder.add_node("handle_editorial_grade", handle_editorial_grade) # Grade

builder.add_node("author_liaison_chat_session", author_liaison_chat_session) # liason between human

# Writing Subgraph
builder.add_node("story_and_dialogue_writers_draft_chapters", story_and_dialogue_writers_draft_chapters) # drafts
builder.add_node("editorial_team_supervisor_review_chapters", editorial_team_supervisor_review_chapters) # Editorial + continuity

builder.add_node("author_relations_brainstorm", author_relations_brainstorm) # new node
builder.add_node("project_initiation", project_initiation)
builder.add_node("market_research", market_research)  # New Node
builder.add_node("contextual_research", contextual_research)  # New Node
builder.add_node("consumer_research", consumer_research) # New Node
builder.add_node("research_supervisor_review", research_supervisor_review)  # New Node
builder.add_node("overall_supervisor_review", overall_supervisor_review) # New Node
builder.add_node("writing_phase", writing_phase)
builder.add_node("editing_phase", editing_phase)
builder.add_node("final_review_phase", final_review_phase)
builder.add_node("publishing_phase", publishing_phase)
builder.add_node("human_validation", human_validation)
builder.add_node("update_story_bible", update_story_bible_node)
builder.add_node("complete", complete) # Complete story.

# Add Edges - Research Phase
builder.add_edge("author_relations_brainstorm", "project_initiation") # new edge
builder.add_edge("project_initiation", "market_research") #new edge
builder.add_edge("market_research", "contextual_research") #new edge
builder.add_edge("contextual_research", "consumer_research") #new edge
builder.add_edge("consumer_research", "research_supervisor_review") #new edge
builder.add_edge("research_supervisor_review", "overall_supervisor_review") #new edge

# Thematics and Narrative subgraph / writing team supervisor review (pre greenlight)
builder.add_edge("overall_supervisor_review", "world_building")
builder.add_edge("world_building", "character_development")
builder.add_edge("character_development", "writing_team_supervisor_review")
builder.add_edge("writing_team_supervisor_review", "editorial_review_initial") # give it to editorial team
builder.add_edge("editorial_review_initial", "handle_editorial_grade") # Editorial review grading

# Editorial logic
builder.add_conditional_edges(
    "handle_editorial_grade",
    handle_editorial_grade_logic,
    {
        "world_building": "world_building",
        "writing_team_supervisor_review": "writing_team_supervisor_review",
        "author_liaison_chat_session": "author_liaison_chat_session"
    }
)

#Start to manuscript
builder.add_edge("author_liaison_chat_session", "story_and_dialogue_writers_draft_chapters") # start chapter drafting

#Writing phase
builder.add_edge("story_and_dialogue_writers_draft_chapters", "editorial_team_supervisor_review_chapters") # start chapter drafting

# Editorial grading, continuity and cohesiveness
builder.add_conditional_edges(
    "editorial_team_supervisor_review_chapters",
    check_all_chapters_written,
    {
        "editorial_team_supervisor_review_chapters": "story_and_dialogue_writers_draft_chapters",
        "complete": "complete" # is done, then complete it.
    }
)
# Complete the story
builder.add_edge("complete", "human_validation")

#Human review.
builder.add_edge("writing_phase", "human_validation")
builder.add_edge("editing_phase", "human_validation")
builder.add_edge("final_review_phase", "human_validation")

# Conditional Edge for Validation
builder.add_conditional_edges(
    "human_validation",
    should_validate,
    {
        "continue": "update_story_bible",
        "human_validation": "human_validation",
    },
)

# Update story bible
builder.add_conditional_edges(
    "update_story_bible",
    has_input,
    {
        "yes": "update_story_bible",  # Loop back to update_story_bible if there's more input.
        "no": {
            "Market Research": "contextual_research",
            "Contextual Research": "consumer_research",
            "Consumer Research": "research_supervisor_review",
            "Research Supervisor Review": "overall_supervisor_review",
            "Overall Supervisor Review": "world_building", # goes to world building.
            "World Building": "character_development",
            "Character Development": "writing_team_supervisor_review",
            "Writing Team Supervisor Review": "editorial_review_initial", # give it to editorial team
            "Editorial Review (Initial)": "handle_editorial_grade", # Editorial review grading
            "Handle Editorial Grade (low)": "world_building",
            "Handle Editorial Grade (middling)": "writing_team_supervisor_review",
            "Handle Editorial Grade (good)": "author_liaison_chat_session",
            "Author Liaison Chat Session": "story_and_dialogue_writers_draft_chapters", # draft chapters
            "Story and Dialogue Writers Draft Chapters": "editorial_team_supervisor_review_chapters",
            "Editorial Team Supervisor Review (Chapters)": "story_and_dialogue_writers_draft_chapters",
            "Writing Phase": "editing_phase",
            "Editing Phase": "final_review_phase",
            "Final Review Phase": "publishing_phase",
            "Human Validation (Outline)": "writing_phase",
            "Human Validation (Sections)": "editing_phase",
            "Human Validation (Edits)": "final_review_phase",
            "Human Validation (Final Story)": "publishing_phase",
            "Awaiting Human Validation: Human Validation (Final Story)": "publishing_phase",
            "Publishing Complete": "complete",
            "Complete": "human_validation"  #Stops the loop.
        }[lambda state: state.current_step](StoryCreationState)
    },
)

# Set the entrypoint
def build_graph() -> StateGraph:
    """
    Builds and returns the configured StateGraph.
    
    Returns:
        StateGraph: The fully configured graph ready for execution
    """
    try:
        # builder is already configured with nodes and edges in the main code
        # Compile the graph and return it
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
        # Create default initial state if none provided
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
                "input": ""
            }

        # Create a config for the graph run
        config = RunnableConfig(
            recursion_limit=100,  # Prevent infinite loops
            tags=["storybook_workflow"]
        )

        # Run the graph
        workflow = graph.configurable_chain()
        result = workflow.invoke(initial_state, config=config)
        
        return result
    except Exception as e:
        print(f"Error running graph: {str(e)}")
        raise

# Running the graph:
graph_ouput = run_graph(graph)
# ----------------------------------------------------------------------
# 6. (Server) Run the Graph
# ----------------------------------------------------------------------

# Example usage (for testing within this file):
if __name__ == "__main__":

    # Initialize the vectorstore (optional, can be done on server start)
    get_story_bible_vectorstore()

    initial_state = StoryCreationState(
        title="The AI Detective",
        genre="Science Fiction Mystery",
        themes=["Artificial Intelligence", "Moral Dilemmas", "Future Crimes"],
    )

    workflow = graph.configurable_chain() # for running on the server

    # Invoke author_relations_brainstorm, project_initiation, and research_phase
    result = workflow.invoke(initial_state.to_dict())

    print(f"Current Step: {result['current_step']}")  # Now should be Market Research

    # Continue the workflow
    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

    next_state = StoryCreationState(**result)
    result = workflow.invoke(next_state.to_dict())
    print(f"Current Step: {result['current_step']}")

# In a real server environment:
# 1.  Use a framework like Flask or FastAPI to create API endpoints.
# 2.  Receive the initial state (or updates) via a POST request.
# 3.  Invoke the `graph.chain()` with the received state.
# 4.  Return the updated state as a JSON response.
