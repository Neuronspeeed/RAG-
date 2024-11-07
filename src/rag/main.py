import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from .loaders import InstructorLoader
from .chunking import EnhancedChunker
from .embeddings import DocumentEmbedder
from .vector_store import EnhancedVectorStore
from .rag import ConversationalRAG
from openai import AsyncOpenAI
import instructor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_rag_system():
    """Initialize the complete RAG system."""
    
    # 1. Load Documents
    logger.info("1. Loading documents...")
    loader = InstructorLoader()
    docs = await loader.load_all_docs()
    logger.info(f"Loaded {len(docs)} documents")
    
    # 2. Process Chunks
    logger.info("\n2. Processing chunks...")
    client = instructor.patch(AsyncOpenAI())
    chunker = EnhancedChunker(client)
    chunks = await chunker.process_chunks(docs)
    logger.info(f"Created {len(chunks['documentation'])} doc chunks and {len(chunks['code'])} code chunks")
    
    # 3. Create Embeddings
    logger.info("\n3. Creating embeddings...")
    embedder = DocumentEmbedder()
    embeddings = embedder.fit_transform(chunks)
    logger.info("Embeddings created successfully")
    
    # 4. Initialize Vector Store
    logger.info("\n4. Initializing vector store...")
    vector_store = EnhancedVectorStore()
    await vector_store.add_documents(chunks)
    logger.info("Vector store initialized")
    
    # 5. Initialize RAG
    logger.info("\n5. Setting up RAG system...")
    rag = ConversationalRAG()
    await rag.initialize(docs)
    logger.info("RAG system ready")
    
    return rag

async def test_queries(rag: ConversationalRAG):
    """Test the RAG system with sample queries."""
    test_queries = [
        "How do I validate API responses with instructor?",
        "Show me examples of using pydantic models",
        "What are the best practices for error handling?",
    ]
    
    logger.info("\nTesting queries:")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        response = await rag.chat(query)
        logger.info(f"Response: {response['response']}")
        logger.info(f"Sources: {response['sources']}")
        logger.info(f"Confidence: {response['confidence']}")

async def main():
    """Main execution function."""
    try:
        # Initialize the system
        rag = await initialize_rag_system()
        
        # Run tests
        await test_queries(rag)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 