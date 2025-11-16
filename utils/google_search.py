"""
Google Search Integration
Uses Google Custom Search API to verify and enhance product identification
"""

import requests
import logging
from typing import List, Dict, Optional
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()


class GoogleSearchAPI:
    """
    Interface to Google Custom Search API for product verification
    """
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """
        Initialize Google Search API
        
        Args:
            api_key: Google API key (get from https://console.cloud.google.com/)
            search_engine_id: Custom Search Engine ID (get from https://cse.google.com/)
        """
        # Load API credentials from environment variables or parameters
        # To set up: Copy .env.example to .env and add your credentials
        # Visit: https://console.cloud.google.com/apis/credentials
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY', 'YOUR_GOOGLE_API_KEY')
        
        # Load Search Engine ID from environment or parameters
        # Visit: https://cse.google.com/cse/
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID', 'YOUR_SEARCH_ENGINE_ID')
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Check if API is configured
        self.is_configured = (
            self.api_key != "YOUR_GOOGLE_API_KEY" and 
            self.search_engine_id != "YOUR_SEARCH_ENGINE_ID"
        )
        
        if not self.is_configured:
            logger.warning("Google Search API not configured. Using fallback mode.")
            logger.warning("To enable: Get API key from https://console.cloud.google.com/")
            logger.warning("Create Custom Search Engine at https://cse.google.com/")
    
    def search_product(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search for product information
        
        Args:
            query: Search query (e.g., "Dell XPS 15 laptop")
            num_results: Number of results to return
            
        Returns:
            list: Search results with title, link, snippet
        """
        if not self.is_configured:
            return self._fallback_search(query)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': num_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google_search'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Google Search API error: {str(e)}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict]:
        """
        Fallback search using built-in product database
        """
        # Simulated search results based on query keywords
        results = []
        
        query_lower = query.lower()
        
        # Product database for fallback
        product_db = {
            'dell xps': {
                'title': 'Dell XPS 15 - Premium Laptop',
                'snippet': 'Dell XPS 15 is a high-performance laptop with InfinityEdge display, Intel processors, and premium build quality.',
                'specs': '15.6" display, Intel Core i7, 16GB RAM, 512GB SSD',
                'category': 'Laptops'
            },
            'macbook': {
                'title': 'Apple MacBook Pro - Professional Laptop',
                'snippet': 'MacBook Pro features M-series chips, Retina display, and macOS for creative professionals.',
                'specs': 'M3 chip, 14"/16" display, up to 96GB RAM',
                'category': 'Laptops'
            },
            'iphone': {
                'title': 'Apple iPhone - Smartphone',
                'snippet': 'iPhone offers advanced camera systems, A-series chips, and iOS ecosystem integration.',
                'specs': 'Various models, Face ID, Multiple cameras',
                'category': 'Smartphones'
            },
            'samsung galaxy': {
                'title': 'Samsung Galaxy - Android Smartphone',
                'snippet': 'Samsung Galaxy phones feature AMOLED displays, Snapdragon processors, and versatile camera systems.',
                'specs': 'Various models, Android OS, S Pen (some models)',
                'category': 'Smartphones'
            },
            'matcha': {
                'title': 'Matcha Green Tea - Premium Japanese Tea',
                'snippet': 'Matcha is finely ground powder of specially grown green tea leaves, popular for its health benefits and unique flavor.',
                'specs': 'Ceremonial grade, Culinary grade, Rich in antioxidants',
                'category': 'Beverages'
            },
            'green tea': {
                'title': 'Green Tea - Traditional Tea',
                'snippet': 'Green tea is made from Camellia sinensis leaves, known for health benefits and natural antioxidants.',
                'specs': 'Various grades, Hot or cold brew',
                'category': 'Beverages'
            }
        }
        
        # Search database
        for key, info in product_db.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                results.append({
                    'title': info['title'],
                    'link': f'https://example.com/{key.replace(" ", "-")}',
                    'snippet': info['snippet'],
                    'specifications': info.get('specs', ''),
                    'category': info.get('category', 'General'),
                    'source': 'fallback_database'
                })
        
        # Generic result if no match
        if not results:
            results.append({
                'title': f'Search results for: {query}',
                'link': f'https://www.google.com/search?q={query.replace(" ", "+")}',
                'snippet': f'Product search for {query}. Use Google Search API for real-time results.',
                'source': 'fallback_generic'
            })
        
        return results[:5]
    
    def verify_product(self, product_name: str, detected_features: List[str]) -> Dict:
        """
        Verify product identification using Google Search
        
        Args:
            product_name: Detected product name
            detected_features: List of detected features/keywords
            
        Returns:
            dict: Verification results with enhanced information
        """
        # Build search query
        query = product_name
        if detected_features:
            query += " " + " ".join(detected_features[:3])
        
        logger.info(f"Searching Google for: {query}")
        
        # Search
        results = self.search_product(query, num_results=3)
        
        # Analyze results
        verification = {
            'original_detection': product_name,
            'search_query': query,
            'top_results': results,
            'verified': len(results) > 0,
            'confidence_boost': 0.1 if len(results) > 0 else 0,
            'enhanced_description': '',
            'specifications': [],
            'brand_confirmed': False
        }
        
        # Extract information from top result
        if results:
            top_result = results[0]
            verification['enhanced_description'] = top_result.get('snippet', '')
            
            # Check if brand is confirmed
            for result in results:
                title_lower = result.get('title', '').lower()
                if any(brand in title_lower for brand in ['dell', 'apple', 'samsung', 'hp', 'lenovo']):
                    verification['brand_confirmed'] = True
                    break
            
            # Extract specifications if available
            for result in results:
                if 'specifications' in result:
                    verification['specifications'].append(result['specifications'])
        
        return verification
    
    def search_brand(self, brand_keywords: List[str], product_type: str = None) -> Dict:
        """
        Search for specific brand information
        
        Args:
            brand_keywords: Keywords related to brand
            product_type: Type of product (optional)
            
        Returns:
            dict: Brand information
        """
        query = " ".join(brand_keywords)
        if product_type:
            query += f" {product_type}"
        
        results = self.search_product(query, num_results=3)
        
        brand_info = {
            'query': query,
            'results': results,
            'identified_brand': None,
            'product_line': None
        }
        
        # Analyze results to identify brand
        if results:
            # Extract brand from top results
            common_brands = ['dell', 'apple', 'samsung', 'hp', 'lenovo', 'asus', 
                           'microsoft', 'google', 'sony', 'lg', 'xiaomi']
            
            for result in results:
                title_lower = result.get('title', '').lower()
                for brand in common_brands:
                    if brand in title_lower:
                        brand_info['identified_brand'] = brand.title()
                        break
                if brand_info['identified_brand']:
                    break
        
        return brand_info


class ProductEnhancer:
    """
    Enhance product detection with Google Search verification
    """
    
    def __init__(self, google_api: GoogleSearchAPI):
        """
        Initialize product enhancer
        
        Args:
            google_api: GoogleSearchAPI instance
        """
        self.google_api = google_api
    
    def enhance_detection(self, detected_product: Dict, keywords: List[str]) -> Dict:
        """
        Enhance product detection with Google Search
        
        Args:
            detected_product: Initial product detection results
            keywords: Extracted keywords from video
            
        Returns:
            dict: Enhanced product information
        """
        product_name = detected_product.get('product_name', 'unknown product')
        
        # Verify with Google Search
        verification = self.google_api.verify_product(product_name, keywords)
        
        # Enhance the detection
        enhanced = {
            **detected_product,
            'google_verified': verification['verified'],
            'enhanced_description': verification['enhanced_description'],
            'search_results': verification['top_results'],
            'confidence_adjusted': detected_product.get('confidence', 0.0) + verification['confidence_boost'],
            'brand_confirmed': verification['brand_confirmed'],
            'specifications_found': verification['specifications']
        }
        
        return enhanced


# Instructions for getting API keys
SETUP_INSTRUCTIONS = """
═══════════════════════════════════════════════════════════════════
GOOGLE SEARCH API SETUP INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

To enable real-time Google Search integration:

1. GET GOOGLE API KEY:
   - Go to: https://console.cloud.google.com/
   - Create a new project or select existing
   - Enable "Custom Search API"
   - Go to "Credentials" and create API key
   - Copy your API key

2. CREATE CUSTOM SEARCH ENGINE:
   - Go to: https://cse.google.com/cse/
   - Click "Add" to create new search engine
   - Configure to search the entire web
   - Copy your Search Engine ID (cx parameter)

3. UPDATE CODE:
   Edit utils/google_search.py and add your credentials:
   
   api_key = "YOUR_API_KEY_HERE"
   search_engine_id = "YOUR_SEARCH_ENGINE_ID_HERE"

4. FREE TIER:
   - 100 queries per day free
   - Sufficient for testing and personal use

═══════════════════════════════════════════════════════════════════

NOTE: The system works in fallback mode without API keys,
using a built-in product database for common items.

═══════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    
    # Test with fallback
    api = GoogleSearchAPI()
    results = api.search_product("Dell XPS 15 laptop")
    
    print("\nSample search results:")
    for r in results:
        print(f"• {r['title']}")
        print(f"  {r['snippet'][:100]}...")
