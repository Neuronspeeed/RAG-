import logging
from typing import Dict, List, Tuple, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
from schemas import DocChunkMetadata, CodeChunkMetadata
import pickle
from metadata_enrichment import MetadataEnricher

logger = logging.getLogger(__name__)

# Configuration for different document types
CHUNK_CONFIG = {
    "code": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "language_hints": {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx"
        }
    },
    "manual": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", " "]
    },
    "text": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", " "]
    }
}

class EnhancedChunker:
    def __init__(self, client, cache_dir: str = "./cache"):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = CHUNK_CONFIG
        self.enricher = MetadataEnricher(client)
        
    def get_document_type(self, doc: Document) -> str:
        """Determine document type and appropriate chunking strategy."""
        source = doc.metadata.get('source', '').lower()
        
        # Check for manual/documentation
        if any(term in source for term in ['manual', 'doc', 'guide', 'tutorial']):
            return "manual"
        
        # Check for code files
        if any(source.endswith(ext) for ext in self.config["code"]["language_hints"].keys()):
            return "code"
            
        return "text"
        
    async def enhance_chunk(self, doc: Document, doc_type: str) -> Document:
        """Enhance a single chunk with metadata."""
        try:
            # Use the MetadataEnricher instance
            enhanced_doc = await self.enricher.enrich_document(doc)
            
            # Add any additional metadata specific to chunking
            enhanced_doc.metadata.update({
                'doc_type': doc_type,
                'is_code': doc_type == "code"
            })
            
            return enhanced_doc
            
        except Exception as e:
            logger.error(f"Enhancement error: {str(e)}")
            doc.metadata.update(self.create_fallback_metadata(doc.page_content, doc_type == "code"))
            return doc

    async def process_chunks(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Process and enhance all chunks."""
        chunked_docs = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            source = doc.metadata.get('source', 'unknown')
            try:
                # Determine document type and chunking strategy
                doc_type = self.get_document_type(doc)
                params = self.config[doc_type]
                
                # Create appropriate splitter
                if doc_type == "code":
                    ext = Path(source).suffix
                    language = params["language_hints"].get(ext)
                    
                    if language:
                        splitter = RecursiveCharacterTextSplitter.from_language(
                            language=language,
                            chunk_size=params["chunk_size"],
                            chunk_overlap=params["chunk_overlap"]
                        )
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=params["chunk_size"],
                            chunk_overlap=params["chunk_overlap"],
                            separators=params.get("separators", ["\n\n", "\n", " "])
                        )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=params["chunk_size"],
                        chunk_overlap=params["chunk_overlap"],
                        separators=params["separators"]
                    )
                
                # Split into chunks
                chunks = splitter.split_text(doc.page_content)
                
                # Process each chunk with metadata
                for i, chunk in enumerate(chunks):
                    chunked_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_number': i,
                            'total_chunks': len(chunks),
                            'doc_type': doc_type,
                            'is_code': doc_type == "code",
                            'is_tutorial': 'tutorial' in source.lower(),
                            'is_documentation': doc_type == "manual",
                            'file_type': Path(source).suffix,
                            'chunk_size': params["chunk_size"],
                            'chunk_overlap': params["chunk_overlap"],
                            'source_file': source,
                            'content_length': len(chunk),
                            'relative_position': i / len(chunks)
                        }
                    )
                    enhanced_doc = await self.enhance_chunk(chunked_doc, doc_type)
                    chunked_docs.append(enhanced_doc)
                    
            except Exception as e:
                logger.error(f"Error processing document {source}: {str(e)}")
                chunked_docs.append(doc)
                continue
        
        # Separate into documentation and code chunks
        doc_chunks = [doc for doc in chunked_docs if doc.metadata['doc_type'] != "code"]
        code_chunks = [doc for doc in chunked_docs if doc.metadata['doc_type'] == "code"]
        
        return {
            "documentation": doc_chunks,
            "code": code_chunks
        }

    async def save_chunks(self, chunks: Dict[str, List[Document]], filename: str = "processed_chunks.pkl"):
        """Save processed chunks to pickle file."""
        filepath = self.cache_dir / filename
        logger.info(f"Saving chunks to {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(chunks, f)
            
        # Log statistics
        total_chunks = len(chunks['documentation']) + len(chunks['code'])
        logger.info(f"Saved {total_chunks} chunks ({len(chunks['documentation'])} docs, {len(chunks['code'])} code)")
        
    @staticmethod
    def load_chunks(filepath: str) -> Dict[str, List[Document]]:
        """Load chunks from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded chunks from {filepath}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks: {str(e)}")
            return {"documentation": [], "code": []}

# Add async main function for testing
async def main():
    from openai import AsyncOpenAI
    import instructor
    from dotenv import load_dotenv
    import os
    from loaders import InstructorLoader
    
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = instructor.patch(AsyncOpenAI())
    
    # Initialize loader and load documents
    loader = InstructorLoader()
    documents = await loader.load_all_docs()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize chunker
    chunker = EnhancedChunker(client)
    
    # Process chunks
    chunks = await chunker.process_chunks(documents)
    
    # Save chunks
    await chunker.save_chunks(chunks)
    
    # Print analysis
    print("\n=== Enhanced Chunk Analysis ===\n")
    print(f"Total Chunks: {len(chunks['documentation']) + len(chunks['code'])}\n")
    
    # Category distribution
    print("Category Distribution:")
    tutorials = sum(1 for doc in chunks['documentation'] if doc.metadata.get('is_tutorial'))
    docs = sum(1 for doc in chunks['documentation'] if doc.metadata.get('is_documentation'))
    code = len(chunks['code'])
    print(f"- tutorials: {tutorials} chunks")
    print(f"- documentation: {docs} chunks")
    print(f"- source_code: {code} chunks\n")
    
    # File type distribution
    file_types = {}
    for doc in chunks['documentation'] + chunks['code']:
        ext = doc.metadata.get('file_type', '')
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print("File Type Distribution:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"- {ext}: {count} chunks")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())