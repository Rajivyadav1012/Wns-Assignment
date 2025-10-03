"""
Web Search Integration
Step 11: Internet search using Serper API
"""
import time
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

from config.settings import settings
from utils.logger import setup_logger
from utils.observability import observability

logger = setup_logger("web_search")

class WebSearchEngine:
    """Web search using Serper API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search engine
        
        Args:
            api_key: Serper API key (uses settings if not provided)
        """
        self.api_key = api_key or settings.SERPER_API_KEY
        self.enabled = bool(self.api_key and "your-" not in self.api_key.lower())
        
        if self.enabled:
            logger.info("Web search enabled with Serper API")
        else:
            logger.warning("Web search disabled - API key not configured")
    
    def search(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict]:
        """
        Search the web using Serper API
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, link, snippet
        """
        if not self.enabled:
            logger.warning("Web search not enabled")
            return []
        
        start_time = time.time()
        
        try:
            url = "https://google.serper.dev/search"
            
            payload = {
                "q": query,
                "num": num_results
            }
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            results = []
            for item in data.get('organic', [])[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'position': item.get('position', 0)
                })
            
            elapsed = time.time() - start_time
            logger.info(f"Web search returned {len(results)} results in {elapsed:.2f}s")
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Web search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web search: {e}")
            return []
    
    def fetch_page_content(self, url: str, max_length: int = 2000) -> str:
        """
        Fetch and extract text content from a webpage
        
        Args:
            url: URL to fetch
            max_length: Maximum content length
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to fetch page content from {url}: {e}")
            return ""
    
    def search_and_fetch(
        self,
        query: str,
        num_results: int = 3,
        fetch_content: bool = True
    ) -> List[Dict]:
        """
        Search and optionally fetch full content
        
        Args:
            query: Search query
            num_results: Number of results
            fetch_content: Whether to fetch full page content
            
        Returns:
            List of results with content
        """
        results = self.search(query, num_results)
        
        if fetch_content:
            for result in results:
                content = self.fetch_page_content(result['link'])
                result['content'] = content
        
        return results

class RAGWithWebSearch:
    """RAG system augmented with web search"""
    
    def __init__(
        self,
        rag_instance,
        enable_web_search: bool = True
    ):
        """
        Initialize RAG with web search
        
        Args:
            rag_instance: Existing RAG instance (Standard/KG/Hybrid)
            enable_web_search: Whether to enable web search
        """
        self.rag = rag_instance
        self.web_search = WebSearchEngine()
        self.enable_web_search = enable_web_search and self.web_search.enabled
        
        # Extract the actual LLM from wrapped instances
        self.llm = self._extract_llm(rag_instance)
        
        logger.info(f"RAG with web search initialized (enabled: {self.enable_web_search})")
    
    def _extract_llm(self, obj):
        """Recursively extract LLM from wrapped RAG instances"""
        if hasattr(obj, 'llm'):
            return obj.llm
        elif hasattr(obj, 'rag'):
            return self._extract_llm(obj.rag)
        elif hasattr(obj, 'rag_instance'):
            return self._extract_llm(obj.rag_instance)
        else:
            return None
    
    def should_use_web_search(
        self,
        rag_results: List[Dict],
        min_results: int = 2,
        min_similarity: float = 0.5
    ) -> bool:
        """
        Determine if web search should be used
        
        Args:
            rag_results: Results from RAG retrieval
            min_results: Minimum number of good results needed
            min_similarity: Minimum similarity threshold
            
        Returns:
            True if web search should be used
        """
        if not self.enable_web_search:
            return False
        
        # Use web search if not enough high-quality results
        good_results = [
            r for r in rag_results
            if r.get('similarity', 0) >= min_similarity
        ]
        
        return len(good_results) < min_results
    
    def format_web_results(self, web_results: List[Dict]) -> str:
        """Format web search results as context"""
        if not web_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(web_results, 1):
            snippet = result.get('snippet', result.get('content', ''))
            context_parts.append(
                f"[Web Result {i}] {result['title']}\n"
                f"Source: {result['link']}\n"
                f"{snippet}"
            )
        
        return "\n\n".join(context_parts)
    
    def query(
        self,
        query: str,
        use_web_search: Optional[bool] = None,
        num_web_results: int = 3,
        **kwargs
    ) -> Dict:
        """
        Query with optional web search fallback
        
        Args:
            query: User query
            use_web_search: Force enable/disable web search (None = auto)
            num_web_results: Number of web results
            **kwargs: Additional arguments for RAG query
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            # First try RAG
            rag_response = self.rag.query(query, **kwargs)
            rag_results = rag_response.get('sources', [])
            
            # Decide if web search is needed
            if use_web_search is None:
                use_web_search = self.should_use_web_search(rag_results)
            
            # Perform web search if needed
            web_results = []
            if use_web_search and self.enable_web_search:
                logger.info("Augmenting with web search...")
                web_results = self.web_search.search(query, num_web_results)
                
                if web_results:
                    # Combine RAG and web results
                    rag_context = self.rag.format_context(rag_results) if hasattr(self.rag, 'format_context') else ""
                    web_context = self.format_web_results(web_results)
                    
                    combined_context = f"{rag_context}\n\n--- Web Search Results ---\n\n{web_context}"
                    
                    # Generate new answer with combined context
                    system_prompt = settings.SYSTEM_PROMPT
                    user_prompt = settings.RAG_PROMPT_TEMPLATE.format(
                        context=combined_context,
                        question=query
                    )
                    
                    answer = self.rag.llm.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 1000)
                    )
                    
                    rag_response['answer'] = answer
                    rag_response['web_results'] = web_results
                    rag_response['used_web_search'] = True
            
            rag_response['total_time'] = time.time() - start_time
            
            return rag_response
            
        except Exception as e:
            logger.error(f"Query with web search failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "web_results": [],
                "error": str(e)
            }

# Test function
def test_web_search():
    """Test web search functionality"""
    print("\n" + "="*70)
    print("TESTING WEB SEARCH")
    print("="*70 + "\n")
    
    # Test basic search
    print("1. Testing Serper API search...")
    search_engine = WebSearchEngine()
    
    if not search_engine.enabled:
        print("   Web search not enabled - add SERPER_API_KEY to .env")
        print("   Get free API key from: https://serper.dev/")
        return
    
    query = "What is RAG in AI?"
    print(f"   Query: {query}\n")
    
    results = search_engine.search(query, num_results=3)
    
    if results:
        print(f"   Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['title']}")
            print(f"      Link: {result['link']}")
            print(f"      Snippet: {result['snippet'][:100]}...\n")
    else:
        print("   No results found\n")
    
    # Test with RAG
    print("\n2. Testing RAG + Web Search integration...")
    try:
        from modules.rag_standard import StandardRAG
        from utils.database import vector_db
        
        stats = vector_db.get_stats()
        if stats['total_vectors'] == 0:
            print("   No documents in database")
            return
        
        rag = StandardRAG("groq", top_k=3, similarity_threshold=0.3)
        rag_with_web = RAGWithWebSearch(rag, enable_web_search=True)
        
        query = "What is the capital of France?"  # Should use web search
        print(f"   Query: {query}\n")
        
        response = rag_with_web.query(query, max_tokens=150)
        
        print(f"   Used web search: {response.get('used_web_search', False)}")
        print(f"   Retrieved from docs: {response.get('retrieved_count', 0)}")
        if response.get('web_results'):
            print(f"   Web results: {len(response['web_results'])}")
        print(f"   Answer: {response['answer'][:200]}...")
        
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n" + "="*70)
    print("Web search tests complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_web_search()