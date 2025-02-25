from typing import Literal, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field

class User(BaseModel):
    """Message from a user."""
    content: str = Field(..., description="The content of the message")
    type: Literal["user", "input"] = Field(..., description="The type of the message")

class BaseMessage(BaseModel):
    """Base class for messages."""

    content: str = Field(..., description="The content of the message")
    type: str = Field(..., description="The type of the message")

class System(BaseMessage):
    """Message from the system."""

    type: Literal["system"] = "system"

class Story(BaseMessage):
    """Message with a story."""

    type: Literal["story"] = "story"

class Image(BaseMessage):
    """Message with an image."""
    # Bytes are not JSON serializable, so we use a string here.
    # If you need to handle actual image bytes, you might need a different approach (e.g., base64 encoding).
    content: Union[str, bytes] = Field(..., description="The content of the image")
    type: Literal["image"] = "image"

class Audio(BaseMessage):
    """Message with audio."""
    #Bytes must be handled not as JSON
    content: Union[str, bytes] = Field(..., description="The content of the audio")
    type: Literal["audio"] = "audio"

def generate_image(image_description: str):
    """placeholder; generate the image and return url"""
    return f"Generated image for: {image_description}"

def generate_story(story: str):
    """placeholder; generate the story so far and assign to state"""
    return f"Generated story so far: {story}"

def system_message():
    """placeholder; generate the system message"""
    return "Once upon a time..."
