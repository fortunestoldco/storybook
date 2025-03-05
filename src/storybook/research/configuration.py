import os
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict

# Copy configuration from RESEARCHER-TO-INTEGRATE/configuration.py
DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:
# ... rest of default structure ...
"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"

# ... rest of enums and Configuration class ...