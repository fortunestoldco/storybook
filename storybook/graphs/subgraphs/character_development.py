# graphs/subgraphs/character_development.py
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import StateGraph

from storybook.agents.character_development import (
    CharacterArcDesignerAgent,
    CharacterRelationshipMapperAgent,
    CharacterResearchAgent,
    DialogueSpecialistAgent,
)
from storybook.agents.project_management import ProjectLeadAgent
from storybook.config import Config
from storybook.utils.state import Character, NovelState, ProjectStatus


class CharacterDevelopmentState(TypedDict):
    """State for the character development subgraph."""

    novel_state: NovelState
    character_profiles: Dict[str, Character]
    character_arcs: Dict[str, Dict[str, Any]]
    transformation_scenes: Dict[str, List[Dict[str, Any]]]
    relationship_map: Dict[str, Dict[str, Any]]
    dialogue_patterns: Dict[str, Dict[str, Any]]
    dialogue_examples: Dict[str, Dict[str, str]]


def create_character_development_graph(config: Config):
    """Create the character development phase subgraph."""
    # Initialize agents
    character_research = CharacterResearchAgent(config)
    character_arc_designer = CharacterArcDesignerAgent(config)
    relationship_mapper = CharacterRelationshipMapperAgent(config)
    dialogue_specialist = DialogueSpecialistAgent(config)
    project_lead = ProjectLeadAgent(config)

    # Define state
    workflow = StateGraph(CharacterDevelopmentState)

    # Define nodes

    # 1. Create character profiles
    def create_character_profiles(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]

        # Determine characters to create based on novel genre and plot needs
        # In a real implementation, this would be more sophisticated
        characters_to_create = [
            {"name": "Protagonist", "role": "Main Character"},
            {"name": "Antagonist", "role": "Opposition"},
            {"name": "Supporting Character 1", "role": "Ally"},
            {"name": "Supporting Character 2", "role": "Mentor"},
        ]

        character_profiles = {}
        for char_info in characters_to_create:
            character = character_research.create_character_profile(
                name=char_info["name"],
                role=char_info["role"],
                genre=novel_state.genre,
                themes=novel_state.themes,
            )
            character_profiles[char_info["name"]] = character

            # Add to novel state
            novel_state.characters[char_info["name"]] = character

        return {
            **state,
            "character_profiles": character_profiles,
            "novel_state": novel_state,
        }

    # 2. Design character arcs
    def design_character_arcs(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        character_profiles = state["character_profiles"]
        character_arcs = {}

        for name, character in character_profiles.items():
            # Refine the character arc based on plot points
            refined_arc = character_arc_designer.refine_character_arc(
                character=character,
                plot_points=novel_state.plot_points,
                themes=novel_state.themes,
            )
            character_arcs[name] = refined_arc

            # Update character in novel state with refined arc
            novel_state.characters[name].arc.update(refined_arc["refined_arc"])

        return {**state, "character_arcs": character_arcs, "novel_state": novel_state}

    # 3. Create transformation scenes
    def create_transformation_scenes(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        character_arcs = state["character_arcs"]
        transformation_scenes = {}

        for name, arc_data in character_arcs.items():
            character = novel_state.characters[name]
            scenes = character_arc_designer.create_transformation_scenes(
                character=character, refined_arc=arc_data["refined_arc"]
            )
            transformation_scenes[name] = scenes

        return {**state, "transformation_scenes": transformation_scenes}

    # 4. Map character relationships
    def map_relationships(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        relationship_map = relationship_mapper.map_relationships(novel_state.characters)

        # Update characters with relationship information
        updated_characters = relationship_mapper.update_character_relationships(
            novel_state.characters, relationship_map
        )

        # Update novel state with the updated characters
        novel_state.characters = updated_characters

        return {
            **state,
            "relationship_map": relationship_map,
            "novel_state": novel_state,
        }

    # 5. Develop dialogue patterns
    def develop_dialogue_patterns(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        dialogue_patterns = {}

        for name, character in novel_state.characters.items():
            refined_patterns = dialogue_specialist.refine_dialogue_patterns(character)
            dialogue_patterns[name] = refined_patterns

            # Update character in novel state with refined dialogue patterns
            novel_state.characters[name].dialogue_patterns.update(
                refined_patterns["refined_dialogue_patterns"]
            )

        return {
            **state,
            "dialogue_patterns": dialogue_patterns,
            "novel_state": novel_state,
        }

    # 6. Generate dialogue examples
    def generate_dialogue_examples(
        state: CharacterDevelopmentState,
    ) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        dialogue_examples = {}

        # Define common situations for dialogue examples
        situations = [
            "Introducing themselves to a stranger",
            "In a moment of high stress or danger",
            "Revealing an important truth",
            "Arguing with someone they care about",
            "Expressing their core motivation",
        ]

        for name, character in novel_state.characters.items():
            examples = dialogue_specialist.generate_dialogue_examples(
                character, situations
            )
            dialogue_examples[name] = examples

        return {**state, "dialogue_examples": dialogue_examples}

    # 7. Set next phase
    def set_next_phase(state: CharacterDevelopmentState) -> CharacterDevelopmentState:
        novel_state = state["novel_state"]
        novel_state = project_lead.set_project_phase(
            novel_state, ProjectStatus.DRAFTING
        )

        # Update completion metrics
        novel_state.phase_progress = 1.0  # This phase is complete

        return {**state, "novel_state": novel_state}

    # Add nodes to graph
    workflow.add_node("create_character_profiles", create_character_profiles)
    workflow.add_node("design_character_arcs", design_character_arcs)
    workflow.add_node("create_transformation_scenes", create_transformation_scenes)
    workflow.add_node("map_relationships", map_relationships)
    workflow.add_node("develop_dialogue_patterns", develop_dialogue_patterns)
    workflow.add_node("generate_dialogue_examples", generate_dialogue_examples)
    workflow.add_node("set_next_phase", set_next_phase)

    # Define edges
    workflow.add_edge("create_character_profiles", "design_character_arcs")
    workflow.add_edge("design_character_arcs", "create_transformation_scenes")
    workflow.add_edge("create_transformation_scenes", "map_relationships")
    workflow.add_edge("map_relationships", "develop_dialogue_patterns")
    workflow.add_edge("develop_dialogue_patterns", "generate_dialogue_examples")
    workflow.add_edge("generate_dialogue_examples", "set_next_phase")

    # Set entry and exit points
    workflow.set_entry_point("create_character_profiles")

    return workflow
