class BaseAgent:
    """Base class for all agents with configurable LLM support."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize agent with optional LLM configuration."""
        if llm_config:
            validate_llm_config(llm_config)
        self.llm = create_llm(llm_config) if llm_config else get_llm()
        self.document_store = DocumentStore()

    def update_llm(self, llm_config: Dict[str, Any]) -> None:
        """Update LLM configuration at runtime."""
        validate_llm_config(llm_config)
        self.llm = create_llm(llm_config)