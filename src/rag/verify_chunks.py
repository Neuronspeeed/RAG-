import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def verify_saved_chunks(filepath: str = "./cache/processed_chunks.pkl"):
    try:
        with open(filepath, 'rb') as f:
            chunks = pickle.load(f)
            
        # Verify structure
        assert 'documentation' in chunks
        assert 'code' in chunks
        
        # Print statistics
        print("\n=== Saved Chunks Verification ===")
        print(f"Documentation chunks: {len(chunks['documentation'])}")
        print(f"Code chunks: {len(chunks['code'])}")
        
        # Verify metadata
        sample_doc = chunks['documentation'][0]
        sample_code = chunks['code'][0]
        
        print("\nMetadata verification:")
        print("Documentation chunk metadata:", list(sample_doc.metadata.keys()))
        print("Code chunk metadata:", list(sample_code.metadata.keys()))
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    verify_saved_chunks() 