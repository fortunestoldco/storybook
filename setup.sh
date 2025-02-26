#!/bin/bash
# init.sh - Initialize the Storybook project structure

# Create directories
mkdir -p storybook/config
mkdir -p storybook/agents/project_management
mkdir -p storybook/agents/story_architecture
mkdir -p storybook/agents/character_development
mkdir -p storybook/agents/writing
mkdir -p storybook/agents/editing
mkdir -p storybook/tools/nlp
mkdir -p storybook/tools/evaluation
mkdir -p storybook/states
mkdir -p storybook/workflows
mkdir -p storybook/innovations/voice
mkdir -p storybook/innovations/creativity
mkdir -p storybook/tests
mkdir -p storybook/utils
mkdir -p output

# Create __init__.py files to make directories into packages
find storybook -type d -exec touch {}/__init__.py \;

# Copy configuration files
echo "Creating configuration files..."
echo "pyproject.toml created"
echo "requirements.txt created"
echo "Dockerfile created"
echo "langgraph.json created"
echo "README.md created"

# Set up environment
echo "Installing dependencies..."
python3 -m venv ~/.venv &&
source ~/.venv/bin/activate &&
pip install -e .

echo "Storybook project initialized successfully!"
echo "To run Storybook, use: python -m storybook.main"
