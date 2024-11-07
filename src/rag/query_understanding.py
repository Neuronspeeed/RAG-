"""Query understanding with chain of thought."""
import instructor
from openai import AsyncOpenAI
from .models import SearchIntent, BackendType
import logging

logger = logging.getLogger(__name__)

async def understand_query(client: AsyncOpenAI, query: str) -> SearchIntent:
    """Extract structured intent with detailed chain of thought."""
    try:
        client = instructor.patch(client)
        return await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=SearchIntent,
            messages=[
                {
                    "role": "system",
                    "content": """You are a query understanding system.
                    
                    Think step by step about:
                    1. What information is being requested
                    2. Which search backends would be most appropriate
                    3. How to rewrite the query for better search
                    4. What temporal context might be relevant
                    
                    Document your complete reasoning in chain_of_thought.
                    Select appropriate backends from: vector, keyword, sql.
                    """
                },
                {"role": "user", "content": query}
            ]
        )
    except Exception as e:
        logger.error(f"Query understanding failed: {str(e)}")
        return SearchIntent(
            original_query=query,
            rewritten_query=query,
            chain_of_thought="Fallback due to error",
            backends=[BackendType.VECTOR]
        )