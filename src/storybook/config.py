import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

class Config:
    """
    Configuration settings for the Storybook LangGraph application.
    This class handles API keys, model settings, data storage, logging,
    and storybook-specific configurations.
    """
    # --- MongoDB Configuration ---
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "storybook")  # Added default
    MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "story_bible")  # Added default
    ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "default")  # Added default

    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo-1106")  # Added default
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # Optional: SerpAPI for search
    ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")  # Optional: ElevenLabs for Text To Speech

    # --- Model Configuration ---
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo-1106")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

    # --- Data Storage ---
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storybook.db")
    STORY_DIRECTORY = os.getenv("STORY_DIRECTORY", "stories")

    # --- Logging Configuration ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Storybook Specific Settings ---
    STORY_LENGTH_WORDS = int(os.getenv("STORY_LENGTH_WORDS", "750"))
    CHAPTER_LENGTH_WORDS = int(os.getenv("CHAPTER_LENGTH_WORDS", "250"))
    STORY_TONE = os.getenv("STORY_TONE", "adventurous")
    STORY_TARGET_AGE = int(os.getenv("STORY_TARGET_AGE", "8"))
    ENABLE_SEARCH = os.getenv("ENABLE_SEARCH", "True").lower() == "true"
    USE_TEXT_TO_SPEECH = os.getenv("USE_TEXT_TO_SPEECH", "False").lower() == "true"

    # --- Advanced LangGraph Settings ---
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "50"))
    EARLY_STOPPING_THRESHOLD = float(os.getenv("EARLY_STOPPING_THRESHOLD", "0.95"))

    # --- Tool Specific Settings ---
    IMAGE_GENERATION_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "dall-e-3")

    @classmethod
    def validate(cls):
        """
        Validate the configuration settings.  Raise an exception if something is missing or invalid.
        """
        required_settings = {
            "MONGODB_URI": "MongoDB URI",
            "MONGODB_DATABASE_NAME": "MongoDB database name",
            "MONGODB_COLLECTION_NAME": "MongoDB collection name",
            "OPENAI_API_KEY": "OpenAI API key",
        }

        for setting, description in required_settings.items():
            if not getattr(cls, setting):
                raise ValueError(f"{description} must be set in environment or have a valid default")

        if cls.STORY_LENGTH_WORDS <= 0:
            raise ValueError("STORY_LENGTH_WORDS must be a positive integer")

        if cls.USE_TEXT_TO_SPEECH and not cls.ELEVEN_LABS_API_KEY:
            raise ValueError("ElevenLabs API key needed for enabling text to speech")

        # Validate MongoDB URI format
        if not cls.MONGODB_URI.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("Invalid MongoDB URI format")

    @classmethod
    def initialize(cls):
        """Initialize and validate the configuration."""
        cls.validate()
        return cls

# Initialize the configuration
config = Config.initialize()

# Export the config instance for use in other modules
__all__ = ['config']
