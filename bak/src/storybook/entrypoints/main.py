"""
Main entry point for the Storybook application.
"""

from storybook.workflows.story_creation import create_story_workflow
from storybook.config import UserRequest


def main():
    # Example user request
    user_request = UserRequest(
        title="Epic Adventure",
        theme="Bravery and Friendship",
        genre="Fantasy",
        target_audience="Young Adults",
        length="Novel",
        keywords=["dragon", "magic", "hero"],
        style="Descriptive",
        special_requirements="Include a wise mentor character",
        user_id="user123",
        deadline="2025-12-31",
        tone="Inspirational",
        story_structure=StoryStructure.HEROS_JOURNEY,
        num_writers=2,
        use_joint_llm=True,
        operation_mode=OperationMode.CREATE,
    )

    # Run the story creation workflow
    story_id = create_story_workflow(user_request)
    print(f"Story created with ID: {story_id}")


if __name__ == "__main__":
    main()
