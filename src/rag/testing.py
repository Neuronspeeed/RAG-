"""Testing utilities for the RAG system."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from instructor import OpenAISchema
from pydantic import Field
from vector_store import ChromaVectorStore
from utils import load_embedded_data, understand_query
from models import QueryContext, SearchResults
from config import settings

logger = logging.getLogger(__name__)

class TestResult(OpenAISchema):
    """Structured test result with reasoning."""
    query: str = Field(description="Original query")
    chain_of_thought: str = Field(description="Reasoning about test execution")
    query_understanding: Dict[str, Any] = Field(description="Query understanding results")
    search_results: Optional[SearchResults] = None
    error: Optional[str] = None
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in test results"
    )
    
    def report(self) -> Dict[str, Any]:
        """Generate detailed test report."""
        report = self.model_dump()
        if hasattr(self, '_raw_response'):
            report["usage"] = self._raw_response.usage.model_dump()
        return report

async def test_query(
    query: str,
    client,
    embedded_data_path: Optional[Path] = None,
    vector_store: Optional[ChromaVectorStore] = None
) -> TestResult:
    """Test query understanding and search functionality with structured output."""
    result = TestResult(
        query=query,
        chain_of_thought="Starting test execution",
        query_understanding={},
        confidence=1.0
    )
    
    try:
        print(f"\nTesting query: '{query}'")
        
        # 1. Initialize vector store with embedded data
        print("\n1. Initializing Vector Store...")
        try:
            if vector_store is None:
                embedded_data = load_embedded_data(
                    embedded_data_path or Path('embedded_data.pkl')
                )
                vector_store = ChromaVectorStore()
                vector_store.initialize_with_data(embedded_data)
                print("Successfully initialized vector store")
                result.chain_of_thought += "\nVector store initialized successfully"
            
        except Exception as e:
            error_msg = f"Vector store initialization error: {str(e)}"
            result.error = error_msg
            result.confidence *= 0.5
            result.chain_of_thought += f"\n{error_msg}"
            return result
        
        # 2. Test query understanding
        print("\n2. Query Understanding:")
        query_context = await understand_query(client, query)
        result.query_understanding = query_context.model_dump()
        result.chain_of_thought += f"\nQuery understood with confidence {query_context.confidence_threshold}"
        
        # 3. Test vector store search
        print("\n3. Search Results:")
        try:
            search_results = await vector_store.search_with_context(query_context)
            result.search_results = search_results
            
            # Print and analyze results
            if search_results.documentation or search_results.code:
                result.chain_of_thought += "\nFound relevant results"
                result.confidence *= search_results.chain_of_thought.confidence
            else:
                result.chain_of_thought += "\nNo results found"
                result.confidence *= 0.7
            
            # Print results for debugging
            _print_results(search_results)
            
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            result.error = error_msg
            result.confidence *= 0.3
            result.chain_of_thought += f"\n{error_msg}"
            
        return result
            
    except Exception as e:
        error_msg = f"Test execution error: {str(e)}"
        result.error = error_msg
        result.confidence = 0.0
        result.chain_of_thought += f"\n{error_msg}"
        return result

def _print_results(results: SearchResults):
    """Helper to print search results."""
    print("\nDocumentation Matches:")
    if not results.documentation:
        print("No documentation matches found")
    else:
        for i, doc in enumerate(results.documentation[:3], 1):
            print(f"\n{i}. Score: {doc.score:.3f}")
            print(f"   Content: {doc.content[:200]}...")
            print(f"   Metadata: {doc.metadata}")
    
    print("\nCode Matches:")
    if not results.code:
        print("No code matches found")
    else:
        for i, code in enumerate(results.code[:3], 1):
            print(f"\n{i}. Score: {code.score:.3f}")
            print(f"   File: {code.metadata.get('source', 'N/A')}")
            print(f"   Content: {code.content[:200]}...")

async def run_tests(
    queries: List[str], 
    client, 
    embedded_data_path: Optional[Path] = None
) -> List[TestResult]:
    """Run multiple test queries with structured output."""
    results = []
    vector_store = None
    
    try:
        embedded_data = load_embedded_data(
            embedded_data_path or Path('embedded_data.pkl')
        )
        vector_store = ChromaVectorStore()
        vector_store.initialize_with_data(embedded_data)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return [TestResult(
            query=query,
            chain_of_thought=f"Failed to initialize vector store: {str(e)}",
            query_understanding={},
            confidence=0.0,
            error=str(e)
        ) for query in queries]
    
    for query in queries:
        result = await test_query(query, client, vector_store=vector_store)
        results.append(result)
        
    return results