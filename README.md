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

## License

MIT
