import pickle
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

def verify_embeddings(filepath: str = "./cache/embedded_data.pkl"):
    """Verify the structure and content of saved embeddings."""
    try:
        logger.info(f"Verifying embeddings from: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Check structure
        assert 'chunks' in data, "Missing chunks"
        assert 'embeddings' in data, "Missing embeddings"
        assert 'vectorizers' in data, "Missing vectorizers"
        
        # Verify chunks
        chunks = data['chunks']
        n_docs = len(chunks['documentation'])
        n_code = len(chunks['code'])
        logger.info(f"\nChunks:")
        logger.info(f"- Documentation: {n_docs}")
        logger.info(f"- Code: {n_code}")
        
        # Verify embeddings
        emb = data['embeddings']
        for doc_type in ['documentation', 'code']:
            for emb_type in ['content', 'metadata']:
                shape = emb[doc_type][emb_type].shape
                logger.info(f"\n{doc_type.title()} {emb_type}:")
                logger.info(f"- Shape: {shape}")
                logger.info(f"- Memory: {emb[doc_type][emb_type].nbytes / 1024 / 1024:.2f} MB")
                
        # Verify vectorizers
        vectorizers = data['vectorizers']
        for name, vec in vectorizers.items():
            vocab_size = len(vec.vocabulary_)
            logger.info(f"\nVectorizer {name}:")
            logger.info(f"- Vocabulary size: {vocab_size}")
            
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_embeddings() 