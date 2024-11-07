"""Document loading and preprocessing pipeline."""
from langchain_community.document_loaders import GitLoader
from tavily import TavilyClient
from langchain.schema import Document
from typing import List, Dict
import logging
from pathlib import Path
import json
from pydantic import BaseModel
from datetime import datetime
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Pydantic model for search results"""
    content: str
    url: str    
    title: str
    score: float

    def to_dict(self):
        return {
            "content": self.content,
            "url": self.url,
            "title": self.title,
            "score": self.score
        }

class InstructorLoader:
    """Enhanced loader for Instructor documentation using Tavily search."""
    
    def __init__(
        self, 
        tavily_api_key: str = None, 
        cache_dir: str = "./tavily_cache",
        force_refresh: bool = False
    ):
        """Initialize with Tavily API key and cache directory."""
        self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found")
            
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.repo_path = Path("./instructor_repo")
        self.cache_dir = Path(cache_dir)
        self.force_refresh = force_refresh
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

    async def load_github_repo(self) -> List[Document]:
        """Load documents from Instructor GitHub repository."""
        logger.info("Loading GitHub repository...")

        try:
            self.repo_path.mkdir(exist_ok=True)
            loader = GitLoader(
                clone_url="https://github.com/instructor-ai/instructor",
                repo_path=str(self.repo_path),
                branch="main",
                file_filter=lambda file_path: file_path.endswith(
                    (".py", ".md", ".rst", ".ipynb")
                )
            )

            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} documents from GitHub")
            return docs

        except Exception as e:
            logger.error(f"Failed to load GitHub repository: {str(e)}")
            return []

    async def tavily_search(self, query: str, cache_filename: str) -> List[SearchResult]:
        """Execute a Tavily search with caching support."""
        cache_path = self.cache_dir / cache_filename

        # Load from cache if available and not force_refresh
        if not self.force_refresh and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_results = json.load(f)
                logger.info(f"Loaded cached Tavily results for '{query}'")
                return [SearchResult(**result) for result in cached_results]

        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_raw_content=True,
                include_images=False
            )
            
            if not isinstance(response, dict) or 'results' not in response:
                logger.error(f"Unexpected response format from Tavily: {response}")
                return []

            results = []
            for result in response['results']:
                try:
                    search_result = SearchResult(
                        content=result.get('content', ''),
                        url=result.get('url', ''),
                        title=result.get('title', ''),
                        score=float(result.get('score', 0.0))
                    )
                    results.append(search_result)
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    continue

            # Cache results
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, ensure_ascii=False)
            logger.info(f"Saved {len(results)} results to cache")

            return results

        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {str(e)}")
            return []

    async def load_web_docs(self) -> List[Document]:
        """Load documents using Tavily search with caching."""
        logger.info("Searching for Instructor documentation...")

        search_queries = [
            "site:python.useinstructor.com",
            "instructor-ai python documentation guide",
            "instructor-ai github documentation",
            "instructor python library tutorial",
            "instructor github examples"
        ]

        all_docs = []
        for query in search_queries:
            logger.info(f"Executing search for: {query}")
            search_results = await self.tavily_search(
                query, 
                cache_filename=f"{query.replace(' ', '_')}.json"
            )

            docs = [
                Document(
                    page_content=result.content,
                    metadata={
                        'source': result.url,
                        'title': result.title,
                        'score': result.score,
                        'type': 'web_doc'
                    }
                )
                for result in search_results
            ]
            all_docs.extend(docs)

        logger.info(f"Successfully loaded {len(all_docs)} documents from web search")
        return all_docs

    async def load_all_docs(self) -> List[Document]:
        """Load both GitHub and web documentation."""
        github_docs = await self.load_github_repo()
        web_docs = await self.load_web_docs()
        
        all_docs = github_docs + web_docs
        logger.info(f"Total documents loaded: {len(all_docs)}")
        
        return all_docs

# Add main execution block for standalone usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        loader = InstructorLoader()
        docs = await loader.load_all_docs()
        print(f"\nTotal Documents: {len(docs)}")
        print("\nSample Sources:")
        for doc in docs[:5]:
            print(f"- {doc.metadata['source']} (Score: {doc.metadata.get('score', 'N/A')})")
        return docs
    
    # Run the async main function
    asyncio.run(main())
