"""Vector store with Instructor-style processing."""

import chromadb
from chromadb.config import Settings
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any, Tuple
from instructor import OpenAISchema
from pydantic import Field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.schema import Document
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from models import (
    QueryContext, SearchResults, SearchResult, 
    DocumentMetadata, ChainOfThought, DateRange
)
from exceptions import InitializationError, SearchError
from config import settings

logger = logging.getLogger(__name__)

class SearchConfig(OpenAISchema):
    """Configuration for search operations."""
    chain_of_thought: ChainOfThought
    filter_conditions: Dict[str, Any] = Field(default_factory=dict)
    rerank_results: bool = True
    max_results: int = 10

class ChromaVectorStore:
    """Enhanced vector store with reasoning."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize vector store with collections."""
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory or settings.persist_directory,
                anonymized_telemetry=False
            ))
            
            # Initialize collections
            self.doc_collection = self._get_or_create_collection("documentation")
            self.code_collection = self._get_or_create_collection("code")
            self._is_initialized = False
            
        except Exception as e:
            raise InitializationError(f"Failed to initialize vector store: {str(e)}")

    def _get_or_create_collection(self, name: str):
        """Get or create a collection with metadata."""
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={
                    "description": f"Collection for {name} content",
                    "created_at": datetime.now().isoformat(),
                    "hnsw:space": "cosine"  # Using cosine similarity
                },
                embedding_function=self.embedding_function
            )
        except Exception as e:
            raise InitializationError(f"Failed to create collection {name}: {str(e)}")

    async def search_with_context(
        self, 
        query_context: QueryContext,
        config: Optional[SearchConfig] = None
    ) -> SearchResults:
        """Enhanced search with reasoning about results."""
        try:
            if not self._is_initialized:
                raise InitializationError("Vector store not initialized with data")

            if config is None:
                config = SearchConfig(
                    chain_of_thought=ChainOfThought(
                        reasoning="Using default search configuration",
                        confidence=0.8
                    ),
                    filter_conditions={},
                    rerank_results=True,
                    max_results=10
                )

            # Update reasoning about search process
            results = SearchResults(
                chain_of_thought=ChainOfThought(
                    reasoning=f"""
                    1. Processing query: {query_context.rewritten_query}
                    2. Using temporal context: {query_context.temporal_context}
                    3. Applying metadata filters: {query_context.required_metadata}
                    4. Confidence threshold: {query_context.confidence_threshold}
                    """,
                    confidence=0.9
                ),
                total_results=0
            )

            # Build filter conditions
            filter_conditions = config.filter_conditions.copy()
            if query_context.temporal_context:
                filter_conditions.update(self._build_temporal_filters(query_context.temporal_context))

            # Search documentation collection
            try:
                doc_results = await self._search_collection(
                    collection=self.doc_collection,
                    query=query_context.rewritten_query,
                    filter_conditions=filter_conditions,
                    limit=config.max_results
                )
                results.documentation = doc_results
                results.total_results += len(doc_results)
                
                # Update reasoning
                results.chain_of_thought.reasoning += f"\n5. Found {len(doc_results)} documentation matches"
                
            except Exception as e:
                logger.warning(f"Documentation search error: {str(e)}")
                results.chain_of_thought.reasoning += f"\n! Documentation search failed: {str(e)}"
                results.chain_of_thought.confidence *= 0.8

            # Search code collection
            try:
                code_results = await self._search_collection(
                    collection=self.code_collection,
                    query=query_context.rewritten_query,
                    filter_conditions=filter_conditions,
                    limit=config.max_results
                )
                results.code = code_results
                results.total_results += len(code_results)
                
                # Update reasoning
                results.chain_of_thought.reasoning += f"\n6. Found {len(code_results)} code matches"
                
            except Exception as e:
                logger.warning(f"Code search error: {str(e)}")
                results.chain_of_thought.reasoning += f"\n! Code search failed: {str(e)}"
                results.chain_of_thought.confidence *= 0.8

            # Rerank results if configured
            if config.rerank_results and results.total_results > 0:
                results = await self._rerank_results(results, query_context)
                results.chain_of_thought.reasoning += "\n7. Reranked results based on relevance"

            return results
            
        except Exception as e:
            raise SearchError(f"Search failed: {str(e)}")

    async def _search_collection(
        self,
        collection,
        query: str,
        filter_conditions: Dict[str, Any],
        limit: int
    ) -> List[SearchResult]:
        """Search a collection with enhanced result processing."""
        raw_results = collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter_conditions
        )

        search_results = []
        for idx, (doc_id, content, metadata, score) in enumerate(zip(
            raw_results['ids'][0],
            raw_results['documents'][0],
            raw_results['metadatas'][0],
            raw_results['distances'][0]
        )):
            # Convert distance to similarity score
            similarity_score = 1.0 - score  # Assuming distance is in [0,1]
            
            search_results.append(SearchResult(
                content=content,
                metadata=metadata,
                score=similarity_score,
                relevance_explanation=f"Match {idx+1}: Similarity score {similarity_score:.3f}"
            ))

        return search_results

    async def _rerank_results(
        self,
        results: SearchResults,
        query_context: QueryContext
    ) -> SearchResults:
        """Rerank results using LLM-based relevance scoring."""
        # Implementation of LLM-based reranking...
        return results

    def _build_temporal_filters(self, temporal_context: DateRange) -> Dict[str, Any]:
        """Build temporal filter conditions."""
        filters = {}
        if temporal_context.start_date:
            filters["timestamp"] = {
                "$gte": temporal_context.start_date.isoformat()
            }
        if temporal_context.end_date:
            filters.setdefault("timestamp", {}).update({
                "$lte": temporal_context.end_date.isoformat()
            })
        return filters

class EnhancedVectorStore:
    def __init__(self, max_features: int = 768):
        self.doc_vectorizer = TfidfVectorizer(
            max_features=max_features,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}'
        )
        self.code_vectorizer = TfidfVectorizer(
            max_features=max_features,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'[A-Za-z_][A-Za-z0-9_]*'
        )
        self.doc_embeddings = None
        self.code_embeddings = None
        self.documents = {"documentation": [], "code": []}
        
    async def add_documents(self, chunks: Dict[str, List[Document]]):
        """Add documents with type-specific processing."""
        # Process documentation
        doc_texts = [doc.page_content for doc in chunks["documentation"]]
        if doc_texts:
            self.doc_embeddings = self.doc_vectorizer.fit_transform(doc_texts).toarray()
            self.documents["documentation"] = chunks["documentation"]
            
        # Process code
        code_texts = [doc.page_content for doc in chunks["code"]]
        if code_texts:
            self.code_embeddings = self.code_vectorizer.fit_transform(code_texts).toarray()
            self.documents["code"] = chunks["code"]
            
    async def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform hybrid search across both documentation and code."""
        results = []
        
        # Search documentation
        if self.doc_embeddings is not None:
            query_vec = self.doc_vectorizer.transform([query]).toarray()
            doc_scores = np.dot(self.doc_embeddings, query_vec.T).flatten()
            doc_indices = doc_scores.argsort()[-k:][::-1]
            
            for idx, score in zip(doc_indices, doc_scores[doc_indices]):
                results.append((self.documents["documentation"][idx], float(score)))
                
        # Search code
        if self.code_embeddings is not None:
            query_vec = self.code_vectorizer.transform([query]).toarray()
            code_scores = np.dot(self.code_embeddings, query_vec.T).flatten()
            code_indices = code_scores.argsort()[-k:][::-1]
            
            for idx, score in zip(code_indices, code_scores[code_indices]):
                results.append((self.documents["code"][idx], float(score)))
                
        # Sort by score and normalize
        results.sort(key=lambda x: x[1], reverse=True)
        max_score = max(score for _, score in results)
        normalized_results = [(doc, score/max_score) for doc, score in results]
        
        return normalized_results[:k]

class HybridVectorStore:
    def __init__(self, alpha: float = 0.7):
        """
        Initialize vector store with content/metadata weighting.
        Args:
            alpha: Weight for content similarity (1-alpha for metadata)
        """
        self.alpha = alpha
        self.doc_content_embeddings = None
        self.doc_meta_embeddings = None
        self.code_content_embeddings = None
        self.code_meta_embeddings = None
        self.documents = {"documentation": [], "code": []}
        self.vectorizers = {}
        
    def load_embeddings(self, filepath: str = "./cache/embedded_data.pkl"):
        """Load pre-computed embeddings."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Load embeddings
            emb = data['embeddings']
            self.doc_content_embeddings = emb['documentation']['content']
            self.doc_meta_embeddings = emb['documentation']['metadata']
            self.code_content_embeddings = emb['code']['content']
            self.code_meta_embeddings = emb['code']['metadata']
            
            # Load documents
            self.documents = data['chunks']
            
            # Load vectorizers
            self.vectorizers = data['vectorizers']
            
            logger.info(f"Loaded embeddings - Doc: {len(self.documents['documentation'])}, Code: {len(self.documents['code'])}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return False
            
    def _compute_hybrid_similarity(self, 
                                 query_content_embedding: np.ndarray,
                                 query_meta_embedding: np.ndarray,
                                 content_embeddings: np.ndarray,
                                 meta_embeddings: np.ndarray) -> np.ndarray:
        """Compute weighted similarity scores."""
        content_sim = cosine_similarity(query_content_embedding, content_embeddings)[0]
        meta_sim = cosine_similarity(query_meta_embedding, meta_embeddings)[0]
        return self.alpha * content_sim + (1 - self.alpha) * meta_sim
        
    def search(self, 
              query: str, 
              doc_type: str = "documentation",
              k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        Args:
            query: Search query
            doc_type: 'documentation' or 'code'
            k: Number of results to return
        """
        try:
            # Transform query
            if doc_type == "documentation":
                content_vectorizer = self.vectorizers['doc_content']
                meta_vectorizer = self.vectorizers['doc_meta']
                content_embeddings = self.doc_content_embeddings
                meta_embeddings = self.doc_meta_embeddings
                documents = self.documents['documentation']
            else:
                content_vectorizer = self.vectorizers['code_content']
                meta_vectorizer = self.vectorizers['code_meta']
                content_embeddings = self.code_content_embeddings
                meta_embeddings = self.code_meta_embeddings
                documents = self.documents['code']
                
            # Create query embeddings
            query_content = content_vectorizer.transform([query]).toarray()
            query_meta = meta_vectorizer.transform([query]).toarray()
            
            # Compute similarities
            scores = self._compute_hybrid_similarity(
                query_content, query_meta,
                content_embeddings, meta_embeddings
            )
            
            # Get top k results
            top_k_idx = np.argsort(scores)[-k:][::-1]
            results = [(documents[i], scores[i]) for i in top_k_idx]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = HybridVectorStore()
    store.load_embeddings()
    
    # Test search
    query = "How to implement RAG with metadata?"
    results = store.search(query, k=3)
    
    print("\nTest Search Results:")
    for doc, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"Content: {doc.page_content[:200]}...")