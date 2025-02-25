import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

class Config:
    """
    Configuration settings for the Storybook LangGraph application.
    """

    # API Keys (replace with your actual API keys or environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # Optional: If using search tool
    # Add other API keys as needed

    # Model Configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo") # Or gpt-4, etc.
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7)) # Adjust for creativity

    # Data Storage (Example - could be a database connection string)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storybook.db")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Storybook Specific Settings
    STORY_LENGTH_WORDS = int(os.getenv("STORY_LENGTH_WORDS", 500))  # Target word count
    STORY_TONE = os.getenv("STORY_TONE", "adventurous") # Default tone
    ENABLE_SEARCH = os.getenv("ENABLE_SEARCH", "True").lower() == "true" #Use Search tool or not.

    # Add more configuration options as needed (e.g., for specific tools, etc.)

    @classmethod
    def validate(cls):
        """
        Validate the configuration settings.  Raise an exception if something is missing or invalid.
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set.")

        # Example: Validate other required settings
        # if not cls.DATABASE_URL:
        #     raise ValueError("DATABASE_URL must be set.")

config = Config()
config.validate()
