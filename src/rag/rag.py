"""RAG implementation for Instructor documentation."""
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import instructor
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
client = instructor.patch(AsyncOpenAI())

class QueryContext(BaseModel):
    """Query context with search parameters."""
    query: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=5, ge=1)
    include_sources: bool = Field(default=True)

class SearchIntent(BaseModel):
    """Model for understanding search intent."""
    query: str
    search_type: str = Field(..., description="Type of search: api, tutorial, concept, or error")
    keywords: List[str] = Field(..., description="Key terms to search for")
    filters: Dict[str, str] = Field(default_factory=dict, description="Optional filters")

class VectorStore:
    """Simple vector store using TF-IDF."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        self.documents: List[Document] = []
        self.vectors = None
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        
    async def search_with_context(
        self, 
        query_context: SearchIntent,
        top_k: int = 5
    ) -> List[Document]:
        """Search documents using TF-IDF similarity."""
        if not self.vectors:
            logger.warning("No documents in vector store")
            return []
            
        # Transform query
        query_vector = self.vectorizer.transform([query_context.query])
        
        # Calculate similarities
        similarities = (self.vectors @ query_vector.T).toarray().flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [self.documents[i] for i in top_indices]
        
        return results

class ConversationalRAG:
    """RAG system with conversation history and error handling."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_history: int = 5,
        documents: Optional[List[Document]] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.conversation_history = []  # Initialize history
        self.vector_store = VectorStore()
        self.client = instructor.patch(AsyncOpenAI())
        
        # Initialize vector store if documents provided
        if documents:
            self.vector_store.add_documents(documents)
            
    async def initialize(self, documents: List[Document]):
        """Initialize or update vector store with documents."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Initialized vector store with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
            
    def _validate_initialization(self) -> bool:
        """Validate system is properly initialized."""
        if not hasattr(self, 'conversation_history'):
            logger.error("Conversation history not initialized")
            return False
        if not self.vector_store or not self.vector_store.documents:
            logger.error("Vector store not initialized or empty")
            return False
        return True
        
    def _get_sources(self, results: List[Document]) -> List[str]:
        """Extract unique sources from search results."""
        return list(set(doc.metadata.get("source", "") for doc in results))
        
    async def chat(self, user_input: str) -> Dict:
        """Process user input and generate response."""
        try:
            # Create query context
            query_context = QueryContext(query=user_input)
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            try:
                # Get relevant documents
                search_results = await self._search(query_context)
                if not search_results:
                    return {
                        "response": "I couldn't find relevant information in the documentation.",
                        "sources": [],
                        "confidence": 0.0
                    }
                
                # Build context from search results
                context = "\n\n".join(doc.page_content for doc in search_results)
                
                # Generate response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specializing in Instructor library. Use the provided context to answer questions accurately."},
                        *self.conversation_history[-self.max_history:],
                        {"role": "system", "content": f"Context from documentation:\n{context}"},
                        {"role": "user", "content": user_input}
                    ]
                )
                
                # Add response to history
                self.conversation_history.append(
                    {"role": "assistant", "content": response.choices[0].message.content}
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "sources": self._get_sources(search_results),
                    "confidence": query_context.confidence_threshold
                }
                
            except Exception as search_error:
                logger.error(f"Search error: {str(search_error)}")
                return {
                    "response": "I encountered an error searching the documentation.",
                    "sources": [],
                    "confidence": 0.0,
                    "error": str(search_error)
                }
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                "response": "I encountered an error processing your request.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            } 

    def _format_search_results(self, results: List[Document]) -> str:
        """Format search results into context string."""
        return "\n\n".join(
            f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
            for doc in results
        )

    async def _search(self, query_context: QueryContext) -> List[Document]:
        """Search documents using vector store."""
        try:
            # Understand query intent
            search_intent = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                response_model=SearchIntent,
                messages=[
                    {"role": "system", "content": "Analyze the search query and extract key information."},
                    {"role": "user", "content": query_context.query}
                ]
            )
            
            # Search documents
            search_results = await self.vector_store.search_with_context(search_intent)
            
            return search_results
        except Exception as search_error:
            logger.error(f"Search error: {str(search_error)}")
            return []