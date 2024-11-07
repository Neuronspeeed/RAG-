import pytest
from typing import List
from langchain.schema import Document
from ..rag import ConversationalRAG
from ..search import SearchError, VectorStore

@pytest.fixture
async def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.md"}),
        Document(page_content="Test content 2", metadata={"source": "test2.md"}),
    ]

@pytest.fixture
async def initialized_rag(sample_documents) -> ConversationalRAG:
    """Create initialized RAG system."""
    rag = ConversationalRAG()
    await rag.initialize(sample_documents)
    return rag

@pytest.mark.asyncio
async def test_rag_initialization(sample_documents):
    """Test RAG system initialization."""
    rag = ConversationalRAG()
    assert hasattr(rag, 'conversation_history')
    assert isinstance(rag.conversation_history, list)
    
    await rag.initialize(sample_documents)
    assert rag.vector_store.documents is not None
    assert len(rag.vector_store.documents) == len(sample_documents)

@pytest.mark.asyncio
async def test_chat_functionality(initialized_rag):
    """Test chat functionality."""
    query = "Test query"
    response = await initialized_rag.chat(query)
    
    assert 'response' in response
    assert 'sources' in response
    assert 'confidence' in response
    assert isinstance(response['sources'], list)
    
@pytest.mark.asyncio
async def test_error_handling(initialized_rag):
    """Test error handling."""
    with pytest.raises(SearchError):
        await initialized_rag.chat("")  # Empty query
        
    # Test invalid initialization
    bad_rag = ConversationalRAG()
    with pytest.raises(Exception):
        await bad_rag.chat("test")  # Not initialized 
        