from setuptools import setup, find_packages

setup(
    name="storybook",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.11",
        "uvicorn>=0.25.0",
        "pydantic>=2.0.0",
        "pymongo>=4.6.0",
        "python-dotenv>=1.0.0",
        "langchain-core>=0.1.0",
        "langchain-anthropic>=0.3.8",
        "langchain-aws>=0.2.14",
        "langchain-mongodb>=0.5.0",
        "langchain-openai>=0.3.7",
        "langchain-ollama>=0.0.1",
        "langgraph>=0.0.20",
        "langsmith>=0.0.30",
        "httpx>=0.26.0",
    ],
    python_requires=">=3.11",
)
