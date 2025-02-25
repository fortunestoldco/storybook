import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

class config:
    """
    Configuration settings for the Storybook LangGraph application.
    This class handles API keys, model settings, data storage, logging,
    and storybook-specific configurations.
    """

    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Required: OpenAI API key
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # Optional: SerpAPI for search
    ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY") #Optional: ElevenLabs for Text To Speech
    # Add other API keys here (e.g., for image generation, other LLMs)

    # --- Model Configuration ---
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo-1106")  # Default LLM
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))  # Creativity (0.0 to 1.0)
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024)) #Maximum number of tokens allowed in output.

    # --- Data Storage ---
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storybook.db")  # SQLite, PostgreSQL, etc.
    STORY_DIRECTORY = os.getenv("STORY_DIRECTORY", "stories") # Directory to save generated stories

    # --- Logging Configuration ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Storybook Specific Settings ---
    STORY_LENGTH_WORDS = int(os.getenv("STORY_LENGTH_WORDS", 750))  # Target word count per story. Increased from previous example.
    CHAPTER_LENGTH_WORDS = int(os.getenv("CHAPTER_LENGTH_WORDS", 250)) #Target word count per chapter
    STORY_TONE = os.getenv("STORY_TONE", "adventurous")  # Default tone/style
    STORY_TARGET_AGE = int(os.getenv("STORY_TARGET_AGE", 8)) #Target audience age.
    ENABLE_SEARCH = os.getenv("ENABLE_SEARCH", "True").lower() == "true" # Use Search tool or not.
    USE_TEXT_TO_SPEECH = os.getenv("USE_TEXT_TO_SPEECH", "False").lower() == "true" #Enable Text to Speech through ElevenLabs

    # --- Advanced LangGraph Settings (Potentially) ---
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 50)) #Max steps for the graph
    EARLY_STOPPING_THRESHOLD = float(os.getenv("EARLY_STOPPING_THRESHOLD", 0.95)) # Example for a reward function.

    # --- Tool Specific Settings ---
    IMAGE_GENERATION_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "dall-e-3") #Model to use for image gen

    # Add more configuration options as needed (e.g., for specific tools, etc.)

    @classmethod
    def validate(cls):
        """
        Validate the configuration settings.  Raise an exception if something is missing or invalid.
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set.")

        if cls.STORY_LENGTH_WORDS <= 0:
            raise ValueError("STORY_LENGTH_WORDS must be a positive integer.")

        # Example: Validate other required settings
        # if not cls.DATABASE_URL:
        #     raise ValueError("DATABASE_URL must be set.")
        if cls.USE_TEXT_TO_SPEECH and not cls.ELEVEN_LABS_API_KEY:
            raise ValueError("ElevenLabs API key needed for enabling text to speech")

config = config()
config.validate()
