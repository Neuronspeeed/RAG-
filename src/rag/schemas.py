from instructor import OpenAISchema
from pydantic import BaseModel, Field
from typing import List, Optional

class DocChunkMetadata(OpenAISchema):
    """Metadata for documentation chunks."""
    topic: str = Field(..., description="Main topic of the chunk")
    summary: str = Field(..., description="Brief summary of the content")
    hypothetical_questions: List[str] = Field(..., description="Potential questions this chunk could answer")
    keywords: List[str] = Field(..., description="Key terms and concepts")
    content_type: str = Field(..., description="Type of content (guide/tutorial/api_reference/example)")
    difficulty_level: str = Field(..., description="Beginner/Intermediate/Advanced")

class CodeChunkMetadata(OpenAISchema):
    """Metadata for code chunks."""
    purpose: str = Field(..., description="Main purpose of the code")
    dependencies: List[str] = Field(..., description="Required imports and dependencies")
    key_concepts: List[str] = Field(..., description="Key programming concepts used")
    complexity: str = Field(..., description="Simple/Moderate/Complex")
    return_type: Optional[str] = Field(None, description="Return type if applicable")
    parameters: Optional[List[str]] = Field(None, description="Input parameters if applicable") 