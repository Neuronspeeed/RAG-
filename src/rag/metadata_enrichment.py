"""Enhanced metadata extraction and enrichment."""
from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from openai import AsyncOpenAI
import logging
from langchain.schema import Document

logger = logging.getLogger(__name__)

class EnhancedMetadata(BaseModel):
    """Enhanced metadata for documents."""
    topic: str = Field(..., description="Main topic of the document")
    summary: str = Field(..., description="Brief summary")
    hypothetical_questions: List[str] = Field(..., description="Potential questions this document answers")
    keywords: List[str] = Field(..., description="Key terms and concepts")
    difficulty: str = Field(..., description="Beginner/Intermediate/Advanced")
    
class MetadataEnricher:
    def __init__(self, client: AsyncOpenAI):
        self.client = instructor.patch(client)
        
    async def enrich_document(self, doc: Document) -> Document:
        """Enrich document with enhanced metadata."""
        try:
            extraction = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=EnhancedMetadata,
                messages=[
                    {"role": "system", "content": "Extract key information from this document."},
                    {"role": "user", "content": doc.page_content}
                ]
            )
            doc.metadata.update(extraction.model_dump())
            return doc
        except Exception as e:
            logger.error(f"Metadata enrichment error: {str(e)}")
            return doc