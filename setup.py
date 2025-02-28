from setuptools import setup, find_packages

setup(
    name="storybook",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain",
        "pydantic",
        "tavily-python",
        "pymongo",
        "openai",
        "anthropic",
        "huggingface_hub",
        "replicate"
    ],
    python_requires=">=3.9",
)