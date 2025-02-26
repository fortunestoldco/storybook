# graphs/subgraphs/revision.py
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import StateGraph

from storybook.agents.editing import (
    DevelopmentalEditorAgent,
    DialogueEnhancementAgent,
    LineEditorAgent,
    TensionOptimizationAgent,
)
from storybook.agents.project_management import ProjectLeadAgent
from storybook.agents.writing import ChapterWriterAgent, WritingSupervisorAgent
from storybook.config import Config
from storybook.utils.state import Chapter, NovelState, ProjectStatus


class RevisionState(TypedDict):
    """State for the revision subgraph."""

    novel_state: NovelState
    structural_analysis: Dict[str, Any]
    revision_priorities: Dict[str, List[str]]
    chapter_revisions: Dict[int, Dict[str, Any]]
    line_edited_chapters: Dict[int, Chapter]
    dialogue_enhanced_chapters: Dict[int, Chapter]
    tension_map: Dict[str, Any]
    tension_optimized_chapters: Dict[int, Chapter]


def create_revision_graph(config: Config):
    """Create the revision phase subgraph."""
    # Initialize agents
    developmental_editor = DevelopmentalEditorAgent(config)
    line_editor = LineEditorAgent(config)
    dialogue_enhancement = DialogueEnhancementAgent(config)
    tension_optimization = TensionOptimizationAgent(config)
    writing_supervisor = WritingSupervisorAgent(config)
    chapter_writer = ChapterWriterAgent(config)
    project_lead = ProjectLeadAgent(config)

    # Define state
    workflow = StateGraph(RevisionState)

    # Define nodes

    # 1. Evaluate structure
    def evaluate_structure(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        structural_analysis = developmental_editor.evaluate_structure(novel_state)

        return {**state, "structural_analysis": structural_analysis}

    # 2. Identify revision priorities
    def identify_revision_priorities(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        revision_priorities = writing_supervisor.coordinate_revisions(novel_state)

        return {**state, "revision_priorities": revision_priorities}

    # 3. Revise chapter structure
    def revise_chapters(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        structural_analysis = state["structural_analysis"]
        revision_priorities = state["revision_priorities"]
        chapter_revisions = {}

        # Get style guide from novel state or use default
        style_guide = {
            "narrative_voice": "Third person limited",
            "tense": "Past tense",
            "tone": "Balanced with moments of levity and gravity",
            "pacing": "Varied with attention to emotional and action beats",
        }

        # Identify chapters that need structural revision
        priority_chapters = []
        for line in revision_priorities.get("priority_revisions", []):
            # Extract chapter numbers from revision priorities
            if "Chapter" in line and any(char.isdigit() for char in line):
                chapter_num = int(
                    "".join(filter(str.isdigit, line.split("Chapter")[1].split()[0]))
                )
                priority_chapters.append(chapter_num)

        # If no priorities identified, revise all chapters
        if not priority_chapters and novel_state.chapters:
            priority_chapters = list(novel_state.chapters.keys())

        for chapter_num in priority_chapters:
            if chapter_num not in novel_state.chapters:
                continue

            chapter = novel_state.chapters[chapter_num]

            # Extract relevant structural notes for this chapter
            structural_notes = ""
            if "sections" in structural_analysis:
                for section_name, section_text in structural_analysis[
                    "sections"
                ].items():
                    if f"Chapter {chapter_num}" in section_text:
                        structural_notes += section_text + "\n"

            # Get chapter-specific revision guidance
            revision_guidance = developmental_editor.revise_chapter_structure(
                chapter=chapter, structural_notes=structural_notes
            )

            # Apply revisions using the chapter writer
            revision_notes = []
            for section_name, section_text in revision_guidance["sections"].items():
                revision_notes.append(
                    f"{section_name.replace('_', ' ').title()}: {section_text}"
                )

            revised_chapter = chapter_writer.revise_chapter(
                chapter=chapter, revision_notes=revision_notes, style_guide=style_guide
            )

            # Update novel state with revised chapter
            novel_state.chapters[chapter_num] = revised_chapter

            # Update word count
            word_count_diff = revised_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff

            chapter_revisions[chapter_num] = {
                "guidance": revision_guidance,
                "revised_chapter": revised_chapter,
            }

        return {
            **state,
            "chapter_revisions": chapter_revisions,
            "novel_state": novel_state,
        }

    # 4. Perform line editing
    def perform_line_editing(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        line_edited_chapters = {}

        for chapter_num, chapter in novel_state.chapters.items():
            edited_chapter = line_editor.edit_chapter(chapter)

            # Update novel state with edited chapter
            novel_state.chapters[chapter_num] = edited_chapter

            # Update word count
            word_count_diff = edited_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff

            # Analyze prose quality
            prose_analysis = line_editor.analyze_prose_quality(edited_chapter.content)
            novel_state.chapters[chapter_num].quality_metrics["prose_quality"] = (
                prose_analysis["overall_score"]
            )

            line_edited_chapters[chapter_num] = edited_chapter

        return {
            **state,
            "line_edited_chapters": line_edited_chapters,
            "novel_state": novel_state,
        }

    # 5. Enhance dialogue
    def enhance_dialogue(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        dialogue_enhanced_chapters = {}

        for chapter_num, chapter in novel_state.chapters.items():
            enhanced_chapter = dialogue_enhancement.enhance_dialogue(
                chapter=chapter, characters=novel_state.characters
            )

            # Update novel state with enhanced chapter
            novel_state.chapters[chapter_num] = enhanced_chapter

            # Update word count
            word_count_diff = enhanced_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff

            # Analyze dialogue quality
            dialogue_analysis = dialogue_enhancement.analyze_dialogue_quality(
                enhanced_chapter.content, novel_state.characters
            )
            novel_state.chapters[chapter_num].quality_metrics["dialogue_quality"] = (
                dialogue_analysis["overall_score"]
            )

            dialogue_enhanced_chapters[chapter_num] = enhanced_chapter

        return {
            **state,
            "dialogue_enhanced_chapters": dialogue_enhanced_chapters,
            "novel_state": novel_state,
        }

    # 6. Create tension map
    def create_tension_map(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        tension_map = tension_optimization.create_tension_map(novel_state)

        return {**state, "tension_map": tension_map}

    # 7. Optimize tension
    def optimize_tension(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]
        tension_map = state["tension_map"]
        tension_optimized_chapters = {}

        tension_values = tension_map.get("tension_values", {})

        for chapter_num, chapter in novel_state.chapters.items():
            # Get target tension for this chapter
            target_tension = tension_values.get(chapter_num, 0.5)

            optimized_chapter = tension_optimization.optimize_tension(
                chapter=chapter, target_tension=target_tension
            )

            # Update novel state with optimized chapter
            novel_state.chapters[chapter_num] = optimized_chapter

            # Update word count
            word_count_diff = optimized_chapter.word_count - chapter.word_count
            novel_state.current_word_count += word_count_diff

            tension_optimized_chapters[chapter_num] = optimized_chapter

        return {
            **state,
            "tension_optimized_chapters": tension_optimized_chapters,
            "novel_state": novel_state,
        }

    # 8. Set next phase
    def set_next_phase(state: RevisionState) -> RevisionState:
        novel_state = state["novel_state"]

        # Increase revision cycle count
        novel_state.revision_cycle += 1

        # Check if we should move to the next phase
        if novel_state.revision_cycle >= config.max_revision_cycles:
            novel_state = project_lead.set_project_phase(
                novel_state, ProjectStatus.OPTIMIZING
            )
            novel_state.phase_progress = 1.0  # This phase is complete
        else:
            # Update progress within the revision phase
            novel_state.phase_progress = (
                novel_state.revision_cycle / config.max_revision_cycles
            )

        return {**state, "novel_state": novel_state}

    # Add nodes to graph
    workflow.add_node("evaluate_structure", evaluate_structure)
    workflow.add_node("identify_revision_priorities", identify_revision_priorities)
    workflow.add_node("revise_chapters", revise_chapters)
    workflow.add_node("perform_line_editing", perform_line_editing)
    workflow.add_node("enhance_dialogue", enhance_dialogue)
    workflow.add_node("create_tension_map", create_tension_map)
    workflow.add_node("optimize_tension", optimize_tension)
    workflow.add_node("set_next_phase", set_next_phase)

    # Define edges
    workflow.add_edge("evaluate_structure", "identify_revision_priorities")
    workflow.add_edge("identify_revision_priorities", "revise_chapters")
    workflow.add_edge("revise_chapters", "perform_line_editing")
    workflow.add_edge("perform_line_editing", "enhance_dialogue")
    workflow.add_edge("enhance_dialogue", "create_tension_map")
    workflow.add_edge("create_tension_map", "optimize_tension")
    workflow.add_edge("optimize_tension", "set_next_phase")

    # Set entry and exit points
    workflow.set_entry_point("evaluate_structure")

    return workflow
