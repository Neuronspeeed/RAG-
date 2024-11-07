# Advanced RAG System with Hybrid Search

A Retrieval Augmented Generation (RAG) system featuring vector store, adaptive chunking, and hybrid search capabilities.

## Features

### Multiple Vector Stores
- ChromaDB-based vector store
- Enhanced vector store with TF-IDF
- Hybrid vector store combining content and metadata

### Advanced Embedding Techniques
- Dual embeddings (content + metadata)
- Adaptive chunking
- Code-specific tokenization
- Documentation-specific processing
- Automatic metadata extraction

### Smart Search
- Hybrid similarity scoring
- Context-aware search
- Temporal filtering
- Result reranking
- Confidence scoring

### Tavily Integration
- Web document search using Tavily
- Caching of search results for performance optimization
- Supports loading Instructor documentation from the web

## Configuration

Create a `.env` file:
```env
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```



## Architecture

1. **Tavily Search Integration**
   - Uses Tavily for web searches
   - Caches search results to reduce API calls
   - Enhances document loading with web-based content

2. **Document Processing**
   - Adaptive chunking
   - Metadata extraction
   - Code/documentation separation

3. **Embedding Generation**
   - TF-IDF vectorization
   - Separate content/metadata embeddings
   - Code-specific tokenization

4. **Search Pipeline**
   - Query understanding
   - Multi-backend search
   - Result synthesis
   - Confidence scoring


## Performance

- Efficient caching of embeddings
- Optimized vector operations
- Parallel processing capabilities
- Memory-efficient data handling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License