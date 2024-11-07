"""Utility functions for the RAG system."""

import logging
from typing import Dict, Union, List, Optional
import pickle
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from instructor import OpenAISchema
from openai import AsyncOpenAI
from exceptions import DataValidationError
from config import settings
import shutil
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DateRange(BaseModel):
    """Date range for temporal queries with chain-of-thought reasoning."""
    chain_of_thought: str = Field(description="Think step by step about the time range")
    start_date: Optional[datetime] = Field(None, description="Start of date range")
    end_date: Optional[datetime] = Field(None, description="End of date range")
    is_relative: bool = Field(default=False, description="Whether this is a relative time range")
    relative_days: Optional[int] = Field(None, description="Number of days relative to now")

class QueryUnderstanding(OpenAISchema):
    """Enhanced query understanding with structured output."""
    rewritten_query: str = Field(
        description="Rewrite the query to be more specific and searchable"
    )
    temporal_context: Optional[DateRange] = Field(
        None,
        description="Temporal aspects of the query"
    )
    domains_to_search: List[str] = Field(
        default=["documentation", "code"],
        description="Which domains to search in"
    )
    required_metadata: List[str] = Field(
        default_factory=list,
        description="Required metadata fields for filtering"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence score for results",
        ge=0.0,
        le=1.0
    )
    
    def report(self):
        """Return a report of the query understanding."""
        dct = self.model_dump()
        if hasattr(self, '_raw_response'):
            dct["usage"] = self._raw_response.usage.model_dump()
        return dct

class DocumentMetadata(OpenAISchema):
    """Enhanced metadata for documents."""
    summary: str = Field(description="Brief summary of the content")
    key_concepts: List[str] = Field(description="Key concepts in the document")
    content_type: str = Field(description="Type of content: documentation or code")
    source: Optional[str] = Field(None, description="Source of the document")
    last_modified: Optional[datetime] = Field(None, description="Last modification date")

def validate_embedded_data(data: Dict) -> bool:
    """Validate and normalize the structure of embedded data."""
    try:
        # Check required keys
        required_keys = {'documentation', 'code'}
        if not all(key in data for key in required_keys):
            raise DataValidationError(f"Missing required keys. Found: {data.keys()}")
        
        # Helper function to normalize document structure
        def normalize_doc(doc, doc_type: str) -> Dict:
            if isinstance(doc, str):
                return {
                    'content': doc,
                    'metadata': DocumentMetadata(
                        summary="",
                        key_concepts=[],
                        content_type=doc_type
                    ).model_dump()
                }
            elif isinstance(doc, dict):
                content = doc.get('content') or doc.get('text') or doc.get('page_content', '')
                metadata = doc.get('metadata', {})
                return {
                    'content': content,
                    'metadata': {**metadata, 'content_type': doc_type}
                }
            raise DataValidationError(f"Invalid document format: {type(doc)}")

        # Normalize documentation
        if isinstance(data['documentation'], dict):
            data['documentation'] = [
                normalize_doc(doc, 'documentation') 
                for doc in data['documentation'].values()
            ]
        elif isinstance(data['documentation'], (list, tuple)):
            data['documentation'] = [
                normalize_doc(doc, 'documentation') 
                for doc in data['documentation']
            ]
        
        # Normalize code
        if isinstance(data['code'], dict):
            data['code'] = [
                normalize_doc(doc, 'code') 
                for doc in data['code'].values()
            ]
        elif isinstance(data['code'], (list, tuple)):
            data['code'] = [
                normalize_doc(doc, 'code') 
                for doc in data['code']
            ]
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        return False

async def understand_query(client: AsyncOpenAI, query: str) -> QueryUnderstanding:
    """Extract query context using Instructor patterns."""
    try:
        return await client.chat.completions.create(
            model=settings.model_name,
            response_model=QueryUnderstanding,
            messages=[{
                "role": "system",
                "content": """You are a query understanding expert. Analyze queries to:
                1. Rewrite them for better search precision
                2. Extract temporal context and date ranges
                3. Identify required metadata fields
                4. Determine appropriate confidence thresholds
                5. Decide which domains to search in"""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.1,
            seed=42
        )
    except Exception as e:
        logger.error(f"Query understanding error: {str(e)}")
        return QueryUnderstanding(
            rewritten_query=query,
            confidence_threshold=settings.default_confidence_threshold
        )

def find_docs(docs: List[Document], search_term: str, limit: int = 5) -> List[Document]:
    """Search for documents containing specific terms."""
    matching_docs = []
    for doc in docs:
        if search_term.lower() in doc.page_content.lower():
            matching_docs.append(doc)
            if len(matching_docs) >= limit:
                break
    return matching_docs

def cleanup_directories():
    """Remove cache and repository directories."""
    paths_to_remove = [
        Path("./instructor_repo"),
        Path("./tavily_cache")
    ]
    
    for path in paths_to_remove:
        if path.exists():
            logger.info(f"Removing {path}")
            shutil.rmtree(path)
            
def analyze_documents(docs: List[Document]) -> Dict:
    """Analyze document collection."""
    analysis = {
        'total_chunks': len(docs),
        'categories': defaultdict(int),
        'file_types': defaultdict(int),
        'source_files': defaultdict(int)
    }
    
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        file_type = Path(source).suffix if source != 'unknown' else 'unknown'
        category = doc.metadata.get('category', 'unknown')
        
        analysis['file_types'][file_type] += 1
        analysis['categories'][category] += 1
        analysis['source_files'][source] += 1
    
    logger.info("\nDocument Analysis:")
    logger.info(f"Total chunks: {analysis['total_chunks']}")
    
    logger.info("\nTop Source Files:")
    for source, count in sorted(
        analysis['source_files'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]:
        logger.info(f"- {source}: {count} chunks")
        
    return analysis

async def test_rag_system():
    """Test RAG system functionality."""
    test_queries = [
        "Show me recent documentation about API validation",
        "Find code examples using pydantic models",
        "What are the best practices for error handling?"
    ]
    
    logger.info("Initializing RAG system...")
    rag = ConversationalRAG()
    
    # Load and analyze documents
    loader = DocumentLoader()
    docs = await loader.load_all_docs()
    analysis = analyze_documents(docs)
    
    # Initialize RAG system
    await rag.initialize(docs)
    
    # Test queries
    logger.info("\nTesting queries:")
    for query in tqdm(test_queries, desc="Processing queries"):
        logger.info(f"\nQuery: {query}")
        response = await rag.chat(query)
        logger.info(f"Response: {response['response']}")
        logger.info(f"Sources: {', '.join(response['sources'])}")
        logger.info(f"Confidence: {response['confidence']}")
        
    return {
        'analysis': analysis,
        'test_results': test_queries
    }