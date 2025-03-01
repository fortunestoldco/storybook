import atexit
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from langchain import graphs

# Add global executor
_browser_executor = None

def _init_browser_executor():
    global _browser_executor
    if _browser_executor is None:
        _browser_executor = ThreadPoolExecutor(max_workers=1)

def _cleanup_browser_executor():
    global _browser_executor
    if _browser_executor:
        _browser_executor.shutdown(wait=False)

def _open_browser(url: str):
    try:
        _init_browser_executor()
        if _browser_executor and not _browser_executor._shutdown:
            _browser_executor.submit(webbrowser.open, url)
    except Exception:
        # Fail silently if browser cannot be opened
        pass

# Register cleanup
atexit.register(_cleanup_browser_executor)

def main():
    # ...existing code...
    if launch_browser:
        _open_browser(f"http://localhost:{port}")
        # Add graph visualization if needed
        # from .plot_graph import create_knowledge_graph, visualize_graph
    # ...existing code...
