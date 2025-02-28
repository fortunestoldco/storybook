from setuptools import setup, find_packages

setup(
    name="storybook",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-community",
        "pydantic",
        "pymongo",
        "openai",
        "anthropic",
        "replicate",
        "tavily-python",
        "huggingface-hub",
        "python-dotenv"
    ]
)