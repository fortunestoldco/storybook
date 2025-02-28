from pathlib import Path

def setup_project_directories():
    """Create necessary project directories."""
    directories = [
        "docs",
        "models",
        "docs/examples",
        "tests/test_agents",
        "storybook/agents"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    setup_project_directories()