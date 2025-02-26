# Storybook: Research and NLP/AI/LM Based Novel Generation System

Storybook is an advanced AI system designed to generate high-quality literary fiction with the potential to achieve best-seller status. It leverages a hierarchical multi-agent architecture, sophisticated NLP tools, and creative innovations to produce engaging and professionally crafted novels.

## Features

- **Hierarchical Multi-Agent System**: Specialized agents for every aspect of novel creation, from market research to character development to final editing
- **MiniLM-Based Quality Analysis**: Self-evaluation and improvement using MiniLM embedding models
- **Adaptive Voice Resonance**: Dynamic system for developing and maintaining a distinctive authorial voice
- **Creative Emergence Facilitator**: System to facilitate emergent creativity across agent collaborations
- **Comprehensive Workflow**: End-to-end process from concept to publication-ready manuscript

## Installation

### Requirements

- Python 3.10 or higher
- OpenAI API key (required)
- Anthropic API key (optional)

### Using pip

```bash
pip install -e .
```

### Using Docker

```bash
docker build -t storybook .
```

## Usage

### Command Line Interface

Run Storybook with default settings:

```bash
python -m storybook.main
```

Specify a configuration file and output directory:

```bash
python -m storybook.main --config path/to/config.json --output path/to/output
```

Use an existing project file to continue work:

```bash
python -m storybook.main --project path/to/project.json
```

### Using Docker

```bash
docker run -v $(pwd)/output:/app/output -e OPENAI_API_KEY=your_key storybook
```

### Using LangGraph CLI

```bash
langgraph-cli run storybook --project_parameters='{"title": "The Dark Secret", "genre": "thriller", "target_audience": "adult", "word_count_target": 80000}'
```

## Configuration

### System Configuration

Create a JSON file with system configuration:

```json
{
  "name": "Storybook Novel Generator",
  "version": "1.0.0",
  "agents": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "model": "gpt-4",
    "self_evaluation_enabled": true,
    "improvement_iterations": 2
  },
  "workflows": {
    "timeout_seconds": 3600,
    "max_iterations_per_phase": 5,
    "quality_thresholds": {
      "draft": 0.65,
      "revision": 0.75,
      "final": 0.85
    }
  }
}
```

### Project Parameters

Create a JSON file with project parameters:

```json
{
  "title": "The Dark Secret",
  "genre": "thriller",
  "target_audience": "adult",
  "word_count_target": 80000,
  "themes": ["betrayal", "redemption", "family secrets"],
  "setting": "Small coastal town in New England"
}
```

## Architecture

Storybook consists of several key components:

1. **Agents**: Specialized LLM-powered agents with specific roles in the novel creation process
2. **Tools**: Utility components including NLP analyzers and evaluation systems
3. **States**: State management for tracking the novel's development
4. **Workflows**: Orchestration of the novel generation process
5. **Innovations**: Advanced systems for voice coherence and creative emergence

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black storybook
isort storybook
```

### Type Checking

```bash
mypy storybook
```

## License

MIT

## Acknowledgments

- LangChain and LangGraph for the agent orchestration framework
- Sentence Transformers library for the MiniLM implementation
- NetworkX for graph-based narrative analysis
```
