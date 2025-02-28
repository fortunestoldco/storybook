# Storybook

A LangGraph-powered workflow for transforming draft manuscripts into finished novels.

## Overview

Storybook uses a sophisticated workflow built with LangGraph to analyze and enhance manuscripts through multiple stages of refinement. The system incorporates market research, content analysis, and targeted improvements across multiple dimensions of storytelling.

## Key Features

- Comprehensive market research and target audience analysis
- Detailed content analysis using NLP techniques
- Character development and dialogue enhancement
- World-building and setting enrichment
- Subplot integration and story arc refinement
- Continuity checking and error correction
- Language polishing and style enhancement
- Final quality review and improvement recommendations

## Architecture

The system is built using:

- LangGraph for workflow orchestration
- LangChain for LLM interactions
- Replicate for NLP tasks
- MongoDB for document storage
- FastAPI for API endpoints

## Usage

1. Upload a manuscript through the API
2. Start the transformation process
3. Monitor progress through each stage
4. Receive the enhanced manuscript with a detailed analysis report

## Development

### Prerequisites

- Python 3.12+
- MongoDB
- Replicate API token
- OpenAI API key (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/storybook.git
cd storybook

# Install dependencies
pip install -e .

# Set environment variables
export MONGODB_URI="mongodb://localhost:27017"
export REPLICATE_API_TOKEN="your_replicate_token"
export OPENAI_API_KEY="your_openai_key"  # Optional

# Run the API server
uvicorn storybook.main:app --reload
```

## Setup Local Models

For local model support (llama.cpp), run:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py
```

## Environment Configuration

1. Copy `.env.example` to `.env`
2. Fill in your API keys and configuration
3. Adjust model paths if needed

## API Endpoints

- POST `/manuscripts`: Upload a new manuscript
- GET `/manuscripts/{manuscript_id}`: Retrieve a manuscript
- POST `/start-transformation`: Start the transformation process
- POST `/transform`: Execute the transformation workflow
- GET `/health`: Health check endpoint

## LLM Configuration

Storybook supports multiple LLM providers:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude-3)
- Replicate (Custom models)
- Ollama (Local models)
- HuggingFace (Custom models)
- LlamaCpp (Local models)

### Example Configuration

```json
{
    "agent_config": {
        "research": {
            "provider": "anthropic",
            "config": {
                "model_name": "claude-3-sonnet",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        },
        "writing": {
            "provider": "openai",
            "config": {
                "model_name": "gpt-4",
                "temperature": 0.9
            }
        },
        "editorial": {
            "provider": "llamacpp",
            "config": {
                "model_path": "./models/llama-2-7b.Q4_K_M.gguf",
                "temperature": 0.3,
                "n_gpu_layers": 1
            }
        }
    }
}
```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- C++ compiler (Visual Studio 2019 Build Tools on Windows, GCC on Linux)
- CMake 3.21 or higher

### Windows Setup

1. Install Visual Studio Build Tools 2019 or later with C++ workload:
   - Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"

2. Create and activate virtual environment:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```cmd
pip install -r requirements-dev.txt
```

### Linux Setup

1. Install build dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake gcc g++

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake gcc gcc-c++
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

## License

MIT
