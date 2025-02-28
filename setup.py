from setuptools import setup, find_packages

setup(
    name="storybook",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "fastapi",
        "pymongo",
        "langgraph",
        "pydantic",
    ],
    python_requires=">=3.9",
)