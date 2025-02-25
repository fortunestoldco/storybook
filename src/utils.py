from typing import Dict, List
from langchain_community.chat_models import ChatOpenAI
from agent.config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
import requests  # For web crawling

# Initialize OpenAI Embeddings (or alternative)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# MongoDB Client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DATABASE_NAME]

def get_llm():
    """Returns a configured ChatOpenAI language model."""
    return ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.7, openai_api_key=OPENAI_API_KEY)

def consolidate_sections(sections: Dict[str, str]) -> str:
    """Consolidates all story sections into a single complete story. Returns the complete story."""
    sorted_sections = dict(sorted(sections.items())) # Sort sections by id for consistent order
    return "\n".join(sorted_sections.values())

def get_story_bible_vectorstore():
    """Retrieves or creates the MongoDB Atlas Vector Search index."""

    # Check if collection exists; create if it doesn't
    if MONGODB_COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(MONGODB_COLLECTION_NAME)
        print(f"Collection '{MONGODB_COLLECTION_NAME}' created.")

    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_URI,
        MONGODB_DATABASE_NAME,
        MONGODB_COLLECTION_NAME,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return vectorstore

def add_to_story_bible(content: str, metadata: Dict = None):
    """Adds content to the story bible in MongoDB."""
    vectorstore = get_story_bible_vectorstore()

    document = Document(page_content=content, metadata=metadata or {})
    vectorstore.add_documents([document])

    print(f"Added to story bible: {content[:50]}...")  # Show first 50 chars

def web_crawl(url: str) -> str:
    """Crawls a webpage and returns the text content."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text  # Or use BeautifulSoup to extract text more cleanly
    except requests.exceptions.RequestException as e:
        print(f"Error crawling {url}: {e}")
        return ""
