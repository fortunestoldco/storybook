# Storybook Langgraph - Deployment Guide

This guide provides instructions for deploying the Storybook Langgraph application on Google Cloud Run with Langsmith tracing enabled.

## Prerequisites

1. Google Cloud account with billing enabled
2. Google Cloud CLI installed and configured
3. MongoDB Atlas account and cluster
4. LangSmith account for tracing
5. OpenAI API key

## Setup Environment Variables

Before deployment, you need to set up the necessary environment variables:

1. Create a `.env.yaml` file for Google Cloud Run:

```yaml
MONGODB_URI: "mongodb+srv://username:password@your-cluster.mongodb.net"
MONGODB_DATABASE_NAME: "storybook"
DOCUMENT_COLLECTION_NAME: "manuscripts"
VECTOR_COLLECTION_NAME: "manuscript_vectors"
OPENAI_API_KEY: "your_openai_api_key"
LANGCHAIN_TRACING_V2: "true"
LANGCHAIN_API_KEY: "your_langchain_api_key"
LANGCHAIN_PROJECT: "storybook"
```

## Build and Deploy

1. **Build the Docker Image:**

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/novel-development
```

2. **Deploy to Google Cloud Run:**

```bash
gcloud run deploy novel-development \
  --image gcr.io/YOUR_PROJECT_ID/novel-development \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --env-vars-file .env.yaml \
  --memory 2Gi
```

## MongoDB Atlas Setup

1. Create a MongoDB Atlas cluster if you don't have one
2. Create a database named `storybook`
3. Create collections: `manuscripts`, `manuscript_vectors`, and `graph_checkpoints`
4. Create an Atlas Search index on the `manuscript_vectors` collection:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

## LangSmith Setup

1. Create a LangSmith account and project
2. Set up API key and environment variables
3. View traces at https://smith.langchain.com/

## Verifying Deployment

After deployment, you can verify that your application is running by accessing the health check endpoint:

```
https://novel-development-HASH.a.run.app/health
```

## Using the API

### 1. Start a Storybook Task

```bash
curl -X POST "https://novel-development-HASH.a.run.app/api/v1/novels" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Novel",
    "author": "Author Name",
    "manuscript": "This is my draft novel...",
    "max_iterations": 3
  }'
```

### 2. Check Task Status

```bash
curl -X GET "https://novel-development-HASH.a.run.app/api/v1/novels/task_1/status"
```

### 3. Retrieve Finished Novel

```bash
curl -X GET "https://novel-development-HASH.a.run.app/api/v1/novels/task_1"
```

### 4. Using the LangServe API

The application also exposes a LangServe API for direct interaction with the graph:

```bash
curl -X POST "https://novel-development-HASH.a.run.app/api/v1/novel-development/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "novel": {
        "title": "My Novel",
        "author": "Author Name",
        "manuscript": "This is my draft novel...",
        "current_stage": "initial"
      },
      "current_agent": "",
      "feedback": "",
      "error": "",
      "completed_agents": [],
      "iterations": 0,
      "max_iterations": 3
    }
  }'
```
