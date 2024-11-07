"""Enhanced embedding functionality with adaptive chunking."""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import pickle
import logging
from langchain.schema import Document
from pathlib import Path
import asyncio
logger = logging.getLogger(__name__)

class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    summary: str = Field(max_length=150)
    key_terms: List[str]
    chunk_type: str = Field(default="documentation")
    source: str
    chunk_index: int

class DocumentEmbedder:
    def __init__(self, max_features=768):
        # Content vectorizers
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
        
        # Metadata vectorizers
        self.doc_meta_vectorizer = TfidfVectorizer(
            max_features=max_features//2,  # Smaller dimension for metadata
            strip_accents='unicode',
            analyzer='word'
        )
        self.code_meta_vectorizer = TfidfVectorizer(
            max_features=max_features//2,
            strip_accents='unicode',
            analyzer='word'
        )
        
    def _get_metadata_text(self, doc: Document) -> str:
        """Combine metadata fields into searchable text."""
        meta = doc.metadata
        fields = [
            meta.get('summary', ''),
            meta.get('topic', ''),
            ' '.join(meta.get('keywords', [])),
            ' '.join(meta.get('hypothetical_questions', [])),
            meta.get('difficulty', '')
        ]
        return ' '.join(filter(None, fields))
        
    def fit_transform(self, chunks: Dict[str, List[Document]]) -> Dict:
        """Create embeddings for both document types"""
        try:
            doc_texts = [doc.page_content for doc in chunks['documentation']]
            code_texts = [doc.page_content for doc in chunks['code']]
            
            print(f"\nProcessing {len(doc_texts)} documentation chunks...")
            doc_content_embeddings = self.doc_vectorizer.fit_transform(doc_texts).toarray()
            doc_meta_texts = [self._get_metadata_text(doc) for doc in chunks['documentation']]
            doc_meta_embeddings = self.doc_meta_vectorizer.fit_transform(doc_meta_texts).toarray()
            
            print(f"\nProcessing {len(code_texts)} code chunks...")
            code_content_embeddings = self.code_vectorizer.fit_transform(code_texts).toarray()
            code_meta_texts = [self._get_metadata_text(doc) for doc in chunks['code']]
            code_meta_embeddings = self.code_meta_vectorizer.fit_transform(code_meta_texts).toarray()
            
            # Store embeddings in documents
            for doc, content_emb, meta_emb in tqdm(
                zip(chunks['documentation'], doc_content_embeddings, doc_meta_embeddings), 
                desc="Storing doc embeddings"):
                doc.metadata['content_embedding'] = content_emb
                doc.metadata['metadata_embedding'] = meta_emb
                
            for doc, content_emb, meta_emb in tqdm(
                zip(chunks['code'], code_content_embeddings, code_meta_embeddings), 
                desc="Storing code embeddings"):
                doc.metadata['content_embedding'] = content_emb
                doc.metadata['metadata_embedding'] = meta_emb
            
            return {
                'documentation': {
                    'content': doc_content_embeddings,
                    'metadata': doc_meta_embeddings
                },
                'code': {
                    'content': code_content_embeddings,
                    'metadata': code_meta_embeddings
                }
            }
            
        except Exception as e:
            logger.error(f"Error in embedding creation: {str(e)}")
            raise

async def process_and_save_embeddings(chunks_path: str = "./cache/processed_chunks.pkl"):
    """Process chunks and save embeddings."""
    try:
        logger.info("=== Starting Embedding Process ===")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
            
        logger.info(f"\nLoaded chunks:")
        logger.info(f"- Documentation: {len(chunks['documentation'])} chunks")
        logger.info(f"- Code: {len(chunks['code'])} chunks")
        
        # Create embeddings
        embedder = DocumentEmbedder()
        embedded_data = embedder.fit_transform(chunks)
        
        # Save embeddings and vectorizers
        output_path = Path(chunks_path).parent / "embedded_data.pkl"
        with open(output_path, 'wb') as f:
            save_data = {
                'chunks': chunks,
                'embeddings': embedded_data,
                'vectorizers': {
                    'doc_content': embedder.doc_vectorizer,
                    'doc_meta': embedder.doc_meta_vectorizer,
                    'code_content': embedder.code_vectorizer,
                    'code_meta': embedder.code_meta_vectorizer
                }
            }
            pickle.dump(save_data, f)
            
        logger.info(f"\nSaved to: {output_path}")
        
        # Print detailed statistics
        logger.info("\n=== Embedding Statistics ===")
        for doc_type in ['documentation', 'code']:
            logger.info(f"\n{doc_type.title()}:")
            for emb_type in ['content', 'metadata']:
                shape = embedded_data[doc_type][emb_type].shape
                logger.info(f"- {emb_type}: {shape[0]} chunks × {shape[1]} dimensions")
                
        # Verify file saved
        file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
        logger.info(f"\nFile size: {file_size:.2f} MB")
        
        logger.info("\n=== Embedding Process Complete ===")
        return embedded_data
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(process_and_save_embeddings())