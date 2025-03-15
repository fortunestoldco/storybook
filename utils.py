import os
import warnings
import torch
import psutil
import gc
import functools
import time
import re
from typing import List, Dict, Any, Optional
import traceback
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)

# Initialize CUDA and handle warnings appropriately
def init_cuda():
    """Initialize CUDA and handle warnings appropriately"""
    try:
        # First, check if PyTorch itself is correctly installed
        logger.info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            try:
                # Print available GPU information
                gpu_count = torch.cuda.device_count()
                logger.info(f"Found {gpu_count} GPU devices:")

                for i in range(gpu_count):
                    try:
                        device_props = torch.cuda.get_device_properties(i)
                        logger.info(f"  GPU {i}: {device_props.name}, {device_props.total_memory / 1e9:.2f} GB memory")
                    except Exception as device_err:
                        logger.warning(f"  GPU {i}: Error getting properties: {str(device_err)}")

                # Try a simple tensor operation to confirm CUDA works
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
                test_result = test_tensor * 2
                logger.info(f"CUDA tensor test successful: {test_result.device}")

                # Set appropriate memory optimization flags if that worked
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled for matrix multiplications where supported")

                # Set more conservative memory settings
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

                # Set PyTorch to release memory more aggressively
                if hasattr(torch.cuda, 'empty_cache'):
                    logger.info("Enabling automatic CUDA cache flushing")
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"CUDA initialization warning (non-fatal): {str(e)}")
                logger.info("Continuing with CPU execution")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
        else:
            logger.info("CUDA not available - using CPU")

    except Exception as e:
        logger.error(f"Error during PyTorch/CUDA setup: {str(e)}")
        logger.info("Continuing with CPU execution")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA

# Monitor memory usage
def get_memory_usage():
    """Get current memory usage statistics"""
    # System memory
    sys_memory = psutil.virtual_memory()

    # GPU memory if available
    gpu_memory = {}
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                gpu_memory[i] = {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i)
                }
        except Exception as e:
            gpu_memory["error"] = str(e)
            logger.warning(f"Error getting GPU memory stats: {str(e)}")

    return {
        "system": {
            "total": sys_memory.total,
            "available": sys_memory.available,
            "percent_used": sys_memory.percent
        },
        "gpu": gpu_memory
    }

# Optimized memory cleanup
def cleanup_memory():
    """Aggressive memory cleanup to prevent leaks"""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

        # Force Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection: collected {collected} objects")

        # Return current memory stats
        memory_stats = get_memory_usage()
        logger.debug(f"After cleanup: System memory: {memory_stats['system']['percent_used']}% used")
        return memory_stats
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        return {
            "system": {"error": str(e), "percent_used": 0},
            "gpu": {"error": str(e)}
        }

# Retry decorator for model invocation
def retry_with_exponential_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {retry + 1}/{max_retries} failed with error: {str(e)}")

                    # Clean up memory after a failure
                    memory_stats = cleanup_memory()
                    logger.info(f"Memory cleanup performed. System memory: {memory_stats['system']['percent_used']}% used")

                    # No delay after the last attempt
                    if retry < max_retries - 1:
                        sleep_time = delay * (backoff_factor ** retry)
                        logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)

            # All retries failed
            logger.error(f"All {max_retries} attempts failed. Last error: {str(last_exception)}")
            raise last_exception

        return wrapper
    return decorator

# Split the manuscript into manageable chunks
def split_manuscript(manuscript: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[Dict[str, Any]]:
    """Split a manuscript into manageable chunks using RecursiveCharacterTextSplitter."""
    if not manuscript:
        logger.warning("Empty manuscript provided to split_manuscript")
        return []

    try:
        # Adjust chunk size based on manuscript length for better performance with large manuscripts
        manuscript_length = len(manuscript)
        if manuscript_length > 100000:  # If over 100K characters
            chunk_size = 2000  # Larger chunks
        if manuscript_length > 500000:  # If over 500K characters
            chunk_size = 5000  # Even larger chunks

        logger.info(f"Splitting manuscript of length {manuscript_length} with chunk size {chunk_size}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        texts = text_splitter.split_text(manuscript)
        logger.info(f"Split manuscript into {len(texts)} initial chunks")

        # Create chunks with metadata
        chunks = []
        for i, text in enumerate(texts):
            chunks.append({
                "chunk_id": i,
                "content": text,
                "start_char": manuscript.find(text),
                "end_char": manuscript.find(text) + len(text),
            })

        # If we have too many chunks (could cause performance issues),
        # combine some adjacent chunks to reduce the total number
        max_chunks = 100
        if len(chunks) > max_chunks:
            logger.info(f"Large manuscript detected with {len(chunks)} chunks. Consolidating to improve performance.")
            consolidated_chunks = []
            for i in range(0, len(chunks), len(chunks) // max_chunks + 1):
                end_idx = min(i + len(chunks) // max_chunks + 1, len(chunks))
                content = "".join([chunk["content"] for chunk in chunks[i:end_idx]])
                consolidated_chunks.append({
                    "chunk_id": len(consolidated_chunks),
                    "content": content,
                    "start_char": chunks[i]["start_char"],
                    "end_char": chunks[end_idx-1]["end_char"] if end_idx <= len(chunks) else chunks[-1]["end_char"],
                    "original_chunks": list(range(i, end_idx))
                })
            chunks = consolidated_chunks
            logger.info(f"Consolidated to {len(chunks)} chunks")

        return chunks
    except Exception as e:
        logger.error(f"Error in split_manuscript: {str(e)}")
        logger.debug(traceback.format_exc())
        return [{
            "chunk_id": 0,
            "content": manuscript[:5000] if manuscript else "",  # Return first 5000 chars as a fallback
            "start_char": 0,
            "end_char": min(5000, len(manuscript)) if manuscript else 0,
            "error": str(e)
        }]

def check_quality_gate(gate_name: str, quality_assessment: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a quality gate is passed."""
    try:
        gates = config.get("quality_gates", {})
        gate_config = gates.get(gate_name, {})

        if not gate_config:
            # If no gate is defined, default to passing
            logger.info(f"No quality gate defined for {gate_name}, defaulting to pass")
            return {"passed": True, "message": f"No quality gate defined for {gate_name}"}

        # Check each criterion in the gate
        passed = True
        reasons = []

        for criterion, threshold in gate_config.items():
            if criterion in quality_assessment:
                value = quality_assessment[criterion]
                if value < threshold:
                    passed = False
                    reasons.append(f"{criterion}: {value} (below threshold {threshold})")
                    logger.info(f"Quality gate {gate_name} criterion failed: {criterion}: {value} < {threshold}")
            else:
                # If the criterion is not in the assessment, consider it failed
                passed = False
                reasons.append(f"{criterion}: not assessed (required)")
                logger.info(f"Quality gate {gate_name} criterion missing: {criterion}")

        if passed:
            logger.info(f"Quality gate {gate_name} passed")
        else:
            logger.info(f"Quality gate {gate_name} failed: {', '.join(reasons)}")

        return {
            "passed": passed,
            "message": "Quality gate passed" if passed else "Quality gate failed",
            "reasons": reasons
        }
    except Exception as e:
        logger.error(f"Error checking quality gate {gate_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "passed": True,  # Default to passing on error to continue workflow
            "message": f"Error checking quality gate: {str(e)}",
            "reasons": [str(e)],
            "error": True
        }

def extract_chunk_references(message: str) -> List[int]:
    """Extract chunk references from a message."""
    if not message:
        return []
        
    try:
        chunk_refs = []

        # Look for patterns like "Chunk 3" or "chunks 4-6"
        chunk_patterns = re.findall(r"[Cc]hunk\s+(\d+)(?:\s*-\s*(\d+))?", message)

        for start, end in chunk_patterns:
            start_idx = int(start)
            if end and end.strip():  # If it's a range
                end_idx = int(end)
                for i in range(start_idx, end_idx + 1):
                    chunk_refs.append(i)
            else:  # If it's a single chunk
                chunk_refs.append(start_idx)

        return chunk_refs
    except Exception as e:
        logger.error(f"Error extracting chunk references: {str(e)}")
        return []

def validate_state(state):
    """
    Validate the state object to ensure it has the necessary components.
    Returns True if valid, False otherwise.
    """
    if not state:
        logger.error("State is None or empty")
        return False
        
    # Check for required keys
    required_keys = ["project", "current_input"]
    for key in required_keys:
        if key not in state:
            logger.error(f"State missing required key: {key}")
            return False
            
    # Check project structure
    project = state.get("project", {})
    project_required_keys = ["id", "title", "manuscript", "manuscript_chunks"]
    for key in project_required_keys:
        if key not in project:
            logger.error(f"Project missing required key: {key}")
            return False
            
    # Check that manuscript_chunks is a list
    if not isinstance(project.get("manuscript_chunks", []), list):
        logger.error("manuscript_chunks is not a list")
        return False
        
    # Check current_input structure
    current_input = state.get("current_input", {})
    if not current_input:
        logger.error("current_input is empty")
        return False
        
    # State is valid
    return True
