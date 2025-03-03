from setuptools import setup, find_packages

setup(
    name="writer2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.11",
        "langchain>=0.3.19",
        "langchain_anthropic>=0.3.8",
        "langchain_aws>=0.2.14",
        "langchain_mongodb>=0.5.0",
        "langchain_openai>=0.3.7",
        "langchain_ollama>=0.0.1",
        "pydantic>=2.10.6",
        "pymongo>=4.11.1",
        "uvicorn>=0.34.0",
        "python-dotenv>=1.0.1",
        "requests>=2.31.0",
        "httpx>=0.26.0"
        "Pyjnius"
    ],
    python_requires=">=3.10",
)
