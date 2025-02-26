# graphs/subgraphs/drafting.py
from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph

from storybook.agents.writing import (
    WritingSupervisorAgent,
    ChapterWriterAgent,
    ContinuityManagerAgent,
    DescriptionSpecialistAgent,
)
from storybook.agents.story_architecture import PlotDevelopmentAgent
from storybook.agents.project_management import ProjectLeadAgent
from storybook.utils.state import NovelState, ProjectStatus, Chapter
from storybook.config import storybookConfig


class DraftingState(TypedDict):
    """State for the drafting subgraph."""

    novel_state: NovelState
    writing_plan: Dict[str, Any]
    chapter_outlines: List[Any]
    drafted_chapters: Dict[int, Chapter]
    continuity_results: Dict[int, Dict[str, Any]]
    enhanced_chapters: Dict[int, Chapter]
    narrative_tracking: Dict[str, Any]


def create_drafting_graph(config: storybookConfig):
    """Create the drafting phase subgraph."""
    # Initialize agents
    writing_supervisor = WritingSupervisorAgent(config)
    chapter_writer = ChapterWriterAgent(config)
    continuity_manager = ContinuityManagerAgent(config)
    description_specialist = DescriptionSpecialistAgent(config)
    plot_development = PlotDevelopmentAgent(config)
    project_lead = ProjectLeadAgent(config)

    # Define state
    workflow = StateGraph(DraftingState)

    # Define nodes

    # 1. Create writing plan
    def create_writing_plan(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]
        writing_plan = writing_supervisor.create_writing_plan(novel_state)

        return {**state, "writing_plan": writing_plan}

    # 2. Create chapter outlines
    def create_chapter_outlines(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]

        # Create detailed chapter outlines based on plot points
        chapter_outlines = plot_development.create_chapter_outlines(
            state=novel_state, plot_points=novel_state.plot_points
        )

        return {**state, "chapter_outlines": chapter_outlines}

    # 3. Draft chapters
    def draft_chapters(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]
        chapter_outlines = state["chapter_outlines"]
        writing_plan = state["writing_plan"]

        # Extract style guide from writing plan
        style_guide = {
            "narrative_voice": "Third person limited",
            "tense": "Past tense",
            "tone": "Balanced with moments of levity and gravity",
            "pacing": "Varied with attention to emotional and action beats",
        }
        if (
            "sections" in writing_plan
            and "stylistic_guidelines" in writing_plan["sections"]
        ):
            # Extract style information from the writing plan
            style_text = writing_plan["sections"]["stylistic_guidelines"]
            for line in style_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    style_guide[key.strip().lower().replace(" ", "_")] = value.strip()

        drafted_chapters = {}
        for chapter_outline in chapter_outlines:
            chapter = chapter_writer.write_chapter(
                chapter_outline=chapter_outline,
                state=novel_state,
                style_guide=style_guide,
            )
            drafted_chapters[chapter.number] = chapter

            # Add to novel state
            novel_state.chapters[chapter.number] = chapter

            # Update word count
            novel_state.current_word_count += chapter.word_count

        return {
            **state,
            "drafted_chapters": drafted_chapters,
            "novel_state": novel_state,
        }

    # 4. Check continuity
    def check_continuity(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]
        drafted_chapters = state["drafted_chapters"]
        continuity_results = {}

        # Get all chapters for context
        all_chapters = list(novel_state.chapters.values())
        all_chapters.sort(key=lambda c: c.number)

        for chapter_num, chapter in drafted_chapters.items():
            # Get previous chapters for continuity checking
            previous_chapters = [c for c in all_chapters if c.number < chapter_num]

            continuity_check = continuity_manager.check_continuity(
                current_chapter=chapter, previous_chapters=previous_chapters
            )
            continuity_results[chapter_num] = continuity_check

            # Update chapter quality metrics
            novel_state.chapters[chapter_num].quality_metrics["continuity"] = (
                continuity_check["consistency_score"]
            )

        return {
            **state,
            "continuity_results": continuity_results,
            "novel_state": novel_state,
        }

    # 5. Enhance descriptions
    def enhance_descriptions(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]
        drafted_chapters = state["drafted_chapters"]
        enhanced_chapters = {}

        for chapter_num, chapter in drafted_chapters.items():
            enhanced_chapter = description_specialist.enhance_descriptions(
                chapter=chapter, settings=novel_state.settings
            )
            enhanced_chapters[chapter_num] = enhanced_chapter

            # Update chapter in novel state
            novel_state.chapters[chapter_num] = enhanced_chapter

            # Update word count difference
            word_count_diff = enhanced_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff

            # Analyze description quality
            quality_analysis = description_specialist.analyze_description_quality(
                enhanced_chapter.content
            )
            novel_state.chapters[chapter_num].quality_metrics["description_quality"] = (
                quality_analysis["overall_score"]
            )

        return {
            **state,
            "enhanced_chapters": enhanced_chapters,
            "novel_state": novel_state,
        }

    # 6. Track narrative elements
    def track_narrative_elements(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]

        # Get all chapters
        all_chapters = list(novel_state.chapters.values())
        all_chapters.sort(key=lambda c: c.number)

        narrative_tracking = continuity_manager.track_narrative_elements(all_chapters)

        return {**state, "narrative_tracking": narrative_tracking}

    # 7. Set next phase
    def set_next_phase(state: DraftingState) -> DraftingState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(
            novel_state, ProjectStatus.REVISING
        )

        # Update completion metrics
        novel_state.phase_progress = 1.0  # This phase is complete

        return {**state, "novel_state": novel_state}

    # Add nodes to graph
    workflow.add_node("create_writing_plan", create_writing_plan)
    workflow.add_node("create_chapter_outlines", create_chapter_outlines)
    workflow.add_node("draft_chapters", draft_chapters)
    workflow.add_node("check_continuity", check_continuity)
    workflow.add_node("enhance_descriptions", enhance_descriptions)
    workflow.add_node("track_narrative_elements", track_narrative_elements)
    workflow.add_node("set_next_phase", set_next_phase)

    # Define edges
    workflow.add_edge("create_writing_plan", "create_chapter_outlines")
    workflow.add_edge("create_chapter_outlines", "draft_chapters")
    workflow.add_edge("draft_chapters", "check_continuity")
    workflow.add_edge("check_continuity", "enhance_descriptions")
    workflow.add_edge("enhance_descriptions", "track_narrative_elements")
    workflow.add_edge("track_narrative_elements", "set_next_phase")

    # Set entry and exit points
    workflow.set_entry_point("create_writing_plan")

    return workflow
