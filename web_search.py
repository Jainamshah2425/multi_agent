from typing import List, Dict, Any
import requests
from duckduckgo_search import DDGS
from config import config
from utils import logger

class WebSearcher:
    def __init__(self):
        self.max_results = config.MAX_WEB_RESULTS
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the web for additional context."""
        if not config.ENABLE_WEB_SEARCH:
            return []
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                
                processed_results = []
                for result in results:
                    processed_results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')
                    })
                
                logger.info(f"Found {len(processed_results)} web results for query: {query}")
                return processed_results
                
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
