from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json  # Remove duplicate import
import re

from langchain_core.prompts import ChatPromptTemplate
from storybook.agents.base import BaseAgent  # Add missing BaseAgent import
