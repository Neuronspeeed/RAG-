"""Custom exceptions for the RAG system."""

class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass

class DataValidationError(RAGError):
    """Raised when embedded data validation fails."""
    pass

class SearchError(RAGError):
    """Raised when search operations fail."""
    pass

class InitializationError(RAGError):
    """Raised when vector store initialization fails."""
    pass 