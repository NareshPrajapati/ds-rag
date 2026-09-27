"""Domain-specific errors so the API layer can map failures to clear responses."""


class RAGError(Exception):
    """Base error for the RAG service."""


class DocumentParsingError(RAGError):
    """Raised when a PDF cannot be parsed into usable text."""


class EmptyCorpusError(RAGError):
    """Raised when a query runs before any document has been ingested."""


class LLMNotConfiguredError(RAGError):
    """Raised when the retrieval agent is used without an LLM configured."""


class AgentError(RAGError):
    """Raised when the agent fails to produce a valid structured answer."""
