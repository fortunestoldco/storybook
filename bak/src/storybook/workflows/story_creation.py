"""
Workflow for story creation.
"""

from storybook.services.ollama_service import OllamaService
from storybook.services.database_service import DatabaseService
from storybook.services.notification_service import NotificationService


def create_story_workflow(user_request):
    # Initialize services
    ollama_service = OllamaService()
    database_service = DatabaseService()
    notification_service = NotificationService()

    # Generate initial story outline
    prompt = user_request.to_prompt_string()
    outline = ollama_service.generate_response(prompt)

    # Store the outline in the database
    story_id = database_service.insert_one("stories", {"outline": outline})

    # Notify the user
    notification_service.send_message(f"Your story outline has been created. Story ID: {story_id}")

    return story_id
