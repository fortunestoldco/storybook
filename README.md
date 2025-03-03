# Novel Writing System with LangGraph

This is a LangGraph-based self-deployed server implementing a comprehensive novel writing system with a hierarchical agent structure.

## System Architecture

The system is structured into strategic and operational levels:

### Strategic Level
- Executive Director Agent (System Controller)
  - Human Feedback Integration Manager
  - Quality Assessment Director
  - Project Timeline Manager

### Operational Level
- Creative Director Agent (with teams for Story Architecture, Character Development, Emotional Engineering)
- Content Development Director Agent (with teams for Research, Writing, Drafting Coordination)
- Editorial Director Agent (with teams for Developmental, Line, and Technical Editing)
- Market Alignment Director Agent (with teams for Cultural Relevance, Reader Experience, Marketing Strategy)

## System Workflow

The workflow progresses through five major phases:
1. **Initialization** - Project setup and initial planning
2. **Development** - Building the foundation of the novel
3. **Creation** - Writing the actual content
4. **Refinement** - Editing and polishing the manuscript
5. **Finalization** - Preparing for publication

## Getting Started

### Prerequisites
- Python 3.10+
- MongoDB
- Docker and Docker Compose (optional)

### Installation

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd novel-writing-system

# Start the containers
docker-compose up -d
```

#### Manual Setup
```bash
# Clone the repository
git clone <repository-url>
cd novel-writing-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MongoDB (must be running separately)
# Then start the server
python server.py
```

## API Usage

The system provides a RESTful API for interaction:

### Create a New Project
```bash
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Novel",
    "genre": "Science Fiction",
    "target_audience": "Young Adult",
    "word_count_target": 80000,
    "description": "A space adventure story"
  }'
```

### Run a Task
```bash
curl -X POST http://localhost:8000/projects/{project_id}/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Develop character profiles",
    "phase": "development"
  }'
```

### Add Human Feedback
```bash
curl -X POST http://localhost:8000/projects/{project_id}/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The protagonist needs a stronger motivation",
    "type": "character",
    "quality_scores": {
      "character_believability": 70,
      "plot_coherence": 85
    }
  }'
```

### Get Project Status
```bash
curl -X GET http://localhost:8000/projects/{project_id}/status
```

### Get Manuscript
```bash
curl -X GET http://localhost:8000/projects/{project_id}/manuscript
```

## Architecture Details

### Components

- **Agents**: Specialized AI components with specific roles
- **Workflows**: Phase-specific graphs defining agent interaction patterns
- **State Management**: Hierarchical state tracking at global, director, and team levels
- **MongoDB Integration**: Persistent storage for all system artifacts
- **Quality Gates**: Criteria that must be met to transition between phases

### Key Features

- **Dynamic Team Activation**: Teams are activated based on manuscript needs
- **Hierarchical Quality Control**: Quality checks at different levels
- **Enhanced Knowledge Persistence**: Specialized knowledge bases
- **Adaptive Resource Allocation**: Computation focused on high-impact areas
- **Human-in-the-Loop Integration**: Strategic placement of human touchpoints

## License

This project is licensed under the MIT License - see the LICENSE file for details.

