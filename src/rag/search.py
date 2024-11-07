"""Search backend implementations."""
from abc import ABC, abstractmethod
from typing import List, Dict
from .models import SearchQuery, SearchResult, BackendType
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class SearchBackend(ABC):
    """Abstract base class for search backends."""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute search with given query."""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to search index."""
        pass

class VectorBackend(SearchBackend):
    """Vector search implementation."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.documents: List[Document] = []
        self.vectors = None
        
    async def add_documents(self, documents: List[Document]) -> None:
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        if not self.vectors:
            return []
            
        query_vector = self.vectorizer.transform([query.rewritten_query])
        similarities = (self.vectors @ query_vector.T).toarray().flatten()
        
        results = []
        for idx, score in enumerate(similarities):
            if query.date_range and not self._in_date_range(self.documents[idx], query.date_range):
                continue
                
            results.append(SearchResult(
                content=self.documents[idx].page_content,
                source=self.documents[idx].metadata.get('source', 'unknown'),
                score=float(score),
                backend=BackendType.VECTOR,
                metadata=self.documents[idx].metadata
            ))
            
        return sorted(results, key=lambda x: x.score, reverse=True)[:5]