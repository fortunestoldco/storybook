class BaseAgent:
    """Base class for all agents with configurable LLM support."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm = create_llm(llm_config) if llm_config else get_llm()
        self.document_store = DocumentStore()