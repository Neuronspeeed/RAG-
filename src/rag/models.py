"""Core models for RAG system."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from langchain.schema import Document

class ChainOfThought(BaseModel):
    """Reasoning steps for search."""
    reasoning: str = Field(description="Step by step reasoning about the search")
    approach: str = Field(description="Chosen search approach")
    
class DocumentMetadata(BaseModel):
    """Enhanced metadata for documents."""
    summary: str
    key_terms: List[str]
    chunk_type: str
    source: str
    chunk_index: int
    content_embedding: Optional[List[float]] = None
    metadata_embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """Single search result with metadata."""
    content: str = Field(description="Document content")
    metadata: DocumentMetadata
    score: float = Field(description="Relevance score")
    
    model_config = dict(arbitrary_types_allowed=True)

class SearchResults(BaseModel):
    """Collection of search results."""
    results: List[SearchResult]
    total_found: int
    query_context: Optional[ChainOfThought] = None
    
    model_config = dict(arbitrary_types_allowed=True)

class QueryContext(BaseModel):
    """Search query context."""
    query: str
    doc_type: str = Field(default="documentation")
    top_k: int = Field(default=5)
    min_score: float = Field(default=0.5)
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = dict(arbitrary_types_allowed=True)

class DateRange(BaseModel):
    """Temporal context for search queries."""
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for temporal filtering"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for temporal filtering"
    )
    relative_range: Optional[str] = Field(
        default=None,
        description="Relative time range (e.g., 'last week', 'past month')"
    )

    def to_filter_dict(self) -> dict:
        """Convert to filter dictionary for vector store."""
        filters = {}
        if self.start_date:
            filters["timestamp__gte"] = self.start_date.isoformat()
        if self.end_date:
            filters["timestamp__lte"] = self.end_date.isoformat()
        return filters