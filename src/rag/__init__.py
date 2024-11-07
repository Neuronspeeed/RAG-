"""RAG (Retrieval Augmented Generation) system for documentation and code search."""

__version__ = "0.1.0"

from .models import TemporalContext, QueryContext
from .vector_store import ChromaVectorStore
from .utils import validate_embedded_data, understand_query
from .exceptions import RAGError, DataValidationError, SearchError

__all__ = [
    'TemporalContext',
    'QueryContext',
    'ChromaVectorStore',
    'validate_embedded_data',
    'understand_query',
    'RAGError',
    'DataValidationError',
    'SearchError'
] 