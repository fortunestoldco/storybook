from __future__ import annotations

# Standard library imports
from typing import Dict, List, Any, Optional
import logging

# Third-party imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel, Field

# Local imports
from storybook.graph import storybook, build_storybook
from storybook.db.document_store import DocumentStore
from storybook.config import HOST, PORT, DEBUG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Storybook",
    version="0.1.0",
    description="A LangGraph-powered workflow for transforming draft manuscripts into finished novels",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ManuscriptInput(BaseModel):
    """Input schema for the manuscript transformation."""

    manuscript_id: str = Field(..., description="ID of the manuscript to transform")
    title: Optional[str] = Field(None, description="Title of the manuscript")
    target_audience: Optional[str] = Field(
        None, description="Target audience (if known)"
    )
    genre: Optional[str] = Field(None, description="Genre of the manuscript (if known)")
    author_notes: Optional[str] = Field(
        None, description="Additional notes from the author"
    )


class ManuscriptOutput(BaseModel):
    """Output schema for the manuscript transformation."""

    manuscript_id: str = Field(..., description="ID of the transformed manuscript")
    status: str = Field(..., description="Status of the transformation")
    final_report: Optional[str] = Field(
        None, description="Final report on the transformed manuscript"
    )
    research_insights: Optional[Dict[str, Any]] = Field(
        None, description="Research insights gathered"
    )
    analysis_results: Optional[Dict[str, Any]] = Field(
        None, description="Analysis results of the manuscript"
    )
    improvement_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Metrics on improvements made"
    )


class ManuscriptUpload(BaseModel):
    title: str = Field(..., description="Title of the manuscript")
    content: str = Field(..., description="Full text content of the manuscript")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata about the manuscript"
    )


class ManuscriptTransformRequest(BaseModel):
    manuscript_id: str = Field(..., description="ID of the manuscript to transform")


class TransformationStatus(BaseModel):
    job_id: str = Field(
        ..., description="Job ID for tracking the transformation process"
    )
    status: str = Field(..., description="Current status of the transformation")
    current_state: Optional[str] = Field(
        None, description="Current state in the workflow"
    )
    message: Optional[str] = Field(None, description="Status message")


# Document store for manuscripts
document_store = DocumentStore()

# Add routes for the novel transformation graph
add_routes(
    app,
    storybook.with_types(input_type=ManuscriptInput, output_type=ManuscriptOutput),
    path="/transform",
)


# API endpoints
@app.post("/manuscripts", response_model=Dict[str, str])
async def upload_manuscript(manuscript: ManuscriptUpload):
    """Upload a new manuscript."""
    try:
        manuscript_id = document_store.store_manuscript(
            manuscript.title, manuscript.content, manuscript.metadata
        )

        # Also store manuscript chunks for vector search
        document_store.store_manuscript_chunks(
            manuscript_id, manuscript.title, manuscript.content
        )

        return {
            "manuscript_id": manuscript_id,
            "message": f"Successfully uploaded manuscript: {manuscript.title}",
        }
    except Exception as e:
        logger.error(f"Error uploading manuscript: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload manuscript: {str(e)}"
        )


@app.get("/manuscripts/{manuscript_id}")
async def get_manuscript(manuscript_id: str):
    """Retrieve a manuscript by ID."""
    try:
        manuscript = document_store.get_manuscript(manuscript_id)
        if not manuscript:
            raise HTTPException(status_code=404, detail="Manuscript not found")

        return manuscript
    except Exception as e:
        logger.error(f"Error retrieving manuscript: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve manuscript: {str(e)}"
        )


@app.post("/start-transformation", response_model=Dict[str, str])
async def start_transformation(request: ManuscriptTransformRequest):
    """Start the transformation process for a manuscript."""
    try:
        # Check if manuscript exists
        manuscript = document_store.get_manuscript(request.manuscript_id)
        if not manuscript:
            raise HTTPException(status_code=404, detail="Manuscript not found")

        # Create a transformation job
        # In a production system, this would initiate an async job
        # For now, we'll just return the manuscript ID

        return {
            "manuscript_id": request.manuscript_id,
            "message": f"Transformation started for manuscript: {manuscript.get('title', 'Untitled')}. "
            f"Use the /transform endpoint with the manuscript_id to run the transformation process.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting transformation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start transformation: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint required by Cloud Run."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT, 
        log_level="debug" if DEBUG else "info"
    )
