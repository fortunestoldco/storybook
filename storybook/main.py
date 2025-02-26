# main.py
import os
import argparse
import json
from typing import Dict, Any, Optional
from datetime import datetime

from storybook.config import Config
from storybook.graphs.main_graph import create_main_graph
from storybook.utils.state import NovelState, ProjectStatus


def save_state(state: Dict[str, Any], output_dir: str) -> None:
    """Save the current state to a file."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert NovelState to dict for serialization
    if "novel_state" in state and state["novel_state"] is not None:
        if isinstance(state["novel_state"], NovelState):
            state_dict = state.copy()
            state_dict["novel_state"] = state["novel_state"].model_dump()

            # Save the state to a JSON file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"novel_state_{timestamp}.json"
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(state_dict, f, indent=2, default=str)

            print(f"State saved to {os.path.join(output_dir, filename)}")

            # If the novel is completed, also save the manuscript
            if state_dict["novel_state"]["status"] == ProjectStatus.COMPLETED.value:
                save_manuscript(state, output_dir)
                save_publication_package(state, output_dir)


def save_manuscript(state: Dict[str, Any], output_dir: str) -> None:
    """Save the completed manuscript to a file."""
    if "novel_state" not in state or state["novel_state"] is None:
        return

    novel_state = state["novel_state"]
    if isinstance(novel_state, NovelState):
        novel_state_dict = novel_state.model_dump()
    else:
        novel_state_dict = novel_state

    # Check if the novel has chapters
    if not novel_state_dict.get("chapters"):
        return

    # Create a directory for the manuscript
    manuscript_dir = os.path.join(output_dir, "manuscript")
    os.makedirs(manuscript_dir, exist_ok=True)

    # Get the project name or use a default
    project_name = novel_state_dict.get("project_name", "Novel")
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Create the full manuscript
    manuscript_path = os.path.join(manuscript_dir, f"{project_name}_{timestamp}.txt")

    with open(manuscript_path, "w") as f:
        # Write the title
        f.write(f"{project_name}\n\n")

        # Sort chapters by number
        chapters = novel_state_dict["chapters"]
        sorted_chapter_nums = sorted([int(num) for num in chapters.keys()])

        # Write each chapter
        for chapter_num in sorted_chapter_nums:
            chapter = chapters[str(chapter_num)]
            f.write(f"CHAPTER {chapter_num}: {chapter['title']}\n\n")
            f.write(f"{chapter['content']}\n\n")

    print(f"Manuscript saved to {manuscript_path}")


def save_publication_package(state: Dict[str, Any], output_dir: str) -> None:
    """Save the publication package to a file."""
    if "publication_package" not in state or state["publication_package"] is None:
        return

    # Create a directory for the publication materials
    publication_dir = os.path.join(output_dir, "publication")
    os.makedirs(publication_dir, exist_ok=True)

    # Get the project name or use a default
    project_name = "Novel"
    if "novel_state" in state and state["novel_state"] is not None:
        novel_state = state["novel_state"]
        if isinstance(novel_state, NovelState):
            project_name = novel_state.project_name
        else:
            project_name = novel_state.get("project_name", "Novel")

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Save the publication package
    package_path = os.path.join(
        publication_dir, f"{project_name}_publication_package_{timestamp}.json"
    )

    with open(package_path, "w") as f:
        json.dump(state["publication_package"], f, indent=2, default=str)

    # Also create a readable version
    readable_path = os.path.join(
        publication_dir, f"{project_name}_publication_package_{timestamp}.txt"
    )

    with open(readable_path, "w") as f:
        package = state["publication_package"]
        f.write(f"TITLE: {package.get('title', 'Untitled')}\n\n")
        f.write(f"BLURB:\n{package.get('blurb', '')}\n\n")
        f.write(f"POSITIONING STATEMENT:\n{package.get('positioning', '')}\n\n")

        f.write("ALTERNATIVE POSITIONING STATEMENTS:\n")
        for i, alt in enumerate(package.get("alternative_positionings", []), 1):
            f.write(f"{i}. {alt}\n")
        f.write("\n")

        f.write("COMPARABLE TITLES:\n")
        for comp in package.get("comp_titles", []):
            f.write(f"- {comp}\n")
        f.write("\n")

        f.write("CATEGORIES:\n")
        categories = package.get("categories", {})
        f.write(f"Primary Category: {categories.get('primary_category', '')}\n")
        f.write("Secondary Categories:\n")
        for cat in categories.get("secondary_categories", []):
            f.write(f"- {cat}\n")
        f.write("\n")

        f.write("KEYWORDS:\n")
        for keyword in package.get("keywords", []):
            f.write(f"- {keyword}\n")
        f.write("\n")

        f.write("MANUSCRIPT DETAILS:\n")
        manuscript = package.get("manuscript", {})
        f.write(f"Word Count: {manuscript.get('total_word_count', 0)}\n")
        f.write(f"Chapter Count: {manuscript.get('chapter_count', 0)}\n")

    print(f"Publication package saved to {package_path} and {readable_path}")


def load_state(input_file: str) -> Optional[Dict[str, Any]]:
    """Load state from a file."""
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return None

    try:
        with open(input_file, "r") as f:
            state_dict = json.load(f)

        # Convert dict to NovelState
        if "novel_state" in state_dict and state_dict["novel_state"] is not None:
            novel_state_dict = state_dict["novel_state"]
            state_dict["novel_state"] = NovelState(**novel_state_dict)

        return state_dict
    except Exception as e:
        print(f"Error loading state: {e}")
        return None


def main(
    config_file: Optional[str] = None,
    input_file: Optional[str] = None,
    output_dir: str = "output",
):
    """Run the novel generation workflow."""
    # Load configuration
    config = Config()
    if config_file:
        # Load custom config if provided
        pass

    # Create the workflow graph
    workflow = create_main_graph(config)

    # Initialize or load state
    initial_state = None
    if input_file:
        initial_state = load_state(input_file)
        if initial_state is None:
            print("Failed to load state, starting with a new state.")

    # Run the workflow
    print("Starting novel generation workflow...")
    state = workflow.invoke(initial_state)

    # Save the final state
    save_state(state, output_dir)

    print("Novel generation completed successfully!")

    # Display summary
    if "novel_state" in state and state["novel_state"] is not None:
        novel_state = state["novel_state"]
        if isinstance(novel_state, NovelState):
            print(f"Project: {novel_state.project_name}")
            print(f"Status: {novel_state.status}")
            print(f"Word Count: {novel_state.current_word_count}")
            print(f"Chapters: {len(novel_state.chapters)}")
        else:
            print(f"Project: {novel_state.get('project_name', 'Unknown')}")
            print(f"Status: {novel_state.get('status', 'Unknown')}")
            print(f"Word Count: {novel_state.get('current_word_count', 0)}")
            print(f"Chapters: {len(novel_state.get('chapters', {}))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a novel using AI agents")
    parser.add_argument("--config", help="Path to a custom configuration file")
    parser.add_argument("--input", help="Path to a previous state file to resume from")
    parser.add_argument(
        "--output", default="output", help="Directory to save output files"
    )

    args = parser.parse_args()

    main(args.config, args.input, args.output)
