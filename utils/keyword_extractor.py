"""
Keyword Extraction and Text Analysis
Extract text from video frames using OCR and analyze for keywords
"""

import cv2
import numpy as np
from typing import List, Dict, Set
import re
import logging
from collections import Counter

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Extract keywords and text from images using OCR
    """
    
    def __init__(self, languages=['en']):
        """
        Initialize the keyword extractor
        
        Args:
            languages (list): List of languages for OCR
        """
        self.languages = languages
        self.reader = None
        
        if EASYOCR_AVAILABLE:
            try:
                logger.info("Initializing EasyOCR...")
                self.reader = easyocr.Reader(languages, gpu=True)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR with GPU: {e}")
                try:
                    self.reader = easyocr.Reader(languages, gpu=False)
                    logger.info("EasyOCR initialized with CPU")
                except Exception as e2:
                    logger.error(f"Failed to initialize EasyOCR: {e2}")
        
        # Tech product keywords
        self.tech_keywords = {
            'brands': [
                'apple', 'samsung', 'dell', 'hp', 'lenovo', 'asus', 'acer', 
                'microsoft', 'sony', 'google', 'nvidia', 'amd', 'intel',
                'xiaomi', 'huawei', 'oneplus', 'razer', 'alienware'
            ],
            'product_types': [
                'laptop', 'notebook', 'ultrabook', 'macbook', 'chromebook',
                'phone', 'smartphone', 'iphone', 'galaxy',
                'tablet', 'ipad', 'watch', 'smartwatch',
                'headphone', 'earbud', 'airpod', 'speaker',
                'camera', 'gopro', 'drone',
                'console', 'playstation', 'xbox', 'switch'
            ],
            'specifications': [
                'gb', 'tb', 'ram', 'ssd', 'hdd', 'storage',
                'core', 'processor', 'cpu', 'gpu', 'graphics',
                'display', 'screen', 'inch', 'resolution', '4k', 'hd', 'uhd',
                'battery', 'mah', 'wireless', 'bluetooth', 'wifi', '5g',
                'camera', 'mp', 'megapixel', 'lens'
            ],
            'model_identifiers': [
                'xps', 'spectre', 'thinkpad', 'ideapad', 'inspiron',
                'macbook pro', 'macbook air', 'imac', 'mac mini',
                'surface', 'pixelbook', 'zenbook', 'vivobook', 'rog'
            ]
        }
        
        # Compile all keywords
        self.all_keywords = set()
        for category in self.tech_keywords.values():
            self.all_keywords.update([k.lower() for k in category])
    
    def extract_text_from_image(self, image) -> List[Dict]:
        """
        Extract text from an image using OCR
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of detected text with bounding boxes and confidence
        """
        if self.reader is None:
            logger.warning("OCR reader not initialized")
            return []
        
        try:
            # Preprocess image for better OCR
            preprocessed = self._preprocess_for_ocr(image)
            
            # Perform OCR
            results = self.reader.readtext(preprocessed)
            
            # Format results
            formatted_results = []
            for bbox, text, confidence in results:
                formatted_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return []
    
    def _preprocess_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    
    def _is_valid_keyword(self, keyword: str) -> bool:
        """
        Check if a keyword is valid (not OCR noise)
        
        Args:
            keyword: Keyword to validate
            
        Returns:
            bool: True if valid, False if noise
        """
        keyword = keyword.strip().lower()
        
        # Filter out single digits or very short meaningless strings
        if len(keyword) <= 1:
            return False
        
        # Filter out pure numbers that are just noise (not specs)
        if keyword.isdigit() and len(keyword) <= 3:
            return False
        
        # Filter out random hex-like strings
        if re.match(r'^[a-z0-9]{1,6}$', keyword) and not any(k in keyword for k in ['xps', 'hd', 'uhd', 'pro', 'air', 'gb', 'tb']):
            # This is likely OCR noise unless it's a known abbreviation
            return False
        
        # Filter out strings with excessive mixed case noise
        if len(re.findall(r'[A-Z]', keyword)) > len(keyword) / 2 and len(keyword) < 10:
            return False
        
        return True
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from text
        
        Args:
            text: Input text
            
        Returns:
            list: List of extracted keywords
        """
        text_lower = text.lower()
        keywords = []
        
        # Extract all matching keywords
        for keyword in self.all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Extract numbers with units (specs, model numbers)
        numbers = re.findall(r'\d+(?:\.\d+)?(?:gb|tb|mah|mp|inch|core|hz|ghz)', text_lower)
        keywords.extend(numbers)
        
        # Extract model numbers (e.g., XPS 15, iPhone 14)
        models = re.findall(r'[a-z]+\s*\d+(?:\s*[a-z]*)?', text_lower)
        keywords.extend(models)
        
        # Filter out invalid keywords
        valid_keywords = [k for k in keywords if self._is_valid_keyword(k)]
        
        return list(set(valid_keywords))
    
    def extract_keywords_from_frames(self, frames: List) -> Dict:
        """
        Extract keywords from multiple video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            dict: Aggregated keyword extraction results
        """
        all_text = []
        all_keywords = []
        high_confidence_text = []
        
        logger.info(f"Extracting text from {len(frames)} frames...")
        
        for i, frame in enumerate(frames):
            # Extract text from frame
            ocr_results = self.extract_text_from_image(frame)
            
            for result in ocr_results:
                text = result['text']
                confidence = result['confidence']
                
                all_text.append(text)
                
                if confidence > 0.5:  # High confidence threshold
                    high_confidence_text.append(text)
                
                # Extract keywords from this text
                keywords = self.extract_keywords(text)
                all_keywords.extend(keywords)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Categorize keywords
        categorized = {
            'brands': [],
            'product_types': [],
            'specifications': [],
            'model_identifiers': [],
            'other': []
        }
        
        for keyword, count in keyword_counts.most_common():
            categorized_flag = False
            for category, keywords_list in self.tech_keywords.items():
                if keyword.lower() in [k.lower() for k in keywords_list]:
                    categorized[category].append({
                        'keyword': keyword,
                        'count': count
                    })
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append({
                    'keyword': keyword,
                    'count': count
                })
        
        # Generate summary
        summary = self._generate_keyword_summary(categorized, high_confidence_text)
        
        return {
            'all_text': all_text,
            'high_confidence_text': high_confidence_text,
            'keyword_counts': dict(keyword_counts.most_common(20)),
            'categorized_keywords': categorized,
            'summary': summary
        }
    
    def _generate_keyword_summary(self, categorized: Dict, text_snippets: List[str]) -> str:
        """Generate a human-readable summary of extracted keywords"""
        summary_parts = []
        
        # Brands
        if categorized['brands']:
            brands = [k['keyword'] for k in categorized['brands'][:3]]
            summary_parts.append(f"Brands mentioned: {', '.join(brands)}")
        
        # Product types
        if categorized['product_types']:
            products = [k['keyword'] for k in categorized['product_types'][:3]]
            summary_parts.append(f"Products: {', '.join(products)}")
        
        # Specifications
        if categorized['specifications']:
            specs = [k['keyword'] for k in categorized['specifications'][:5]]
            summary_parts.append(f"Specifications: {', '.join(specs)}")
        
        # Model identifiers
        if categorized['model_identifiers']:
            models = [k['keyword'] for k in categorized['model_identifiers'][:3]]
            summary_parts.append(f"Models: {', '.join(models)}")
        
        if not summary_parts:
            summary_parts.append("No significant keywords detected")
        
        return ". ".join(summary_parts)
    
    def identify_product_from_text(self, text: str) -> Dict:
        """
        Identify product information from text
        
        Args:
            text: Input text
            
        Returns:
            dict: Product identification results
        """
        text_lower = text.lower()
        
        result = {
            'brand': None,
            'product_type': None,
            'model': None,
            'specifications': [],
            'full_description': None
        }
        
        # Identify brand
        for brand in self.tech_keywords['brands']:
            if brand in text_lower:
                result['brand'] = brand.title()
                break
        
        # Identify product type
        for product_type in self.tech_keywords['product_types']:
            if product_type in text_lower:
                result['product_type'] = product_type.title()
                break
        
        # Identify model
        for model in self.tech_keywords['model_identifiers']:
            if model in text_lower:
                result['model'] = model.upper()
                break
        
        # Extract specifications
        for spec in self.tech_keywords['specifications']:
            if spec in text_lower:
                result['specifications'].append(spec)
        
        # Generate full description
        if result['brand'] or result['product_type'] or result['model']:
            parts = [p for p in [result['brand'], result['model'], result['product_type']] if p]
            result['full_description'] = ' '.join(parts)
        
        return result


class ContentAnalyzer:
    """
    Analyze video content and generate descriptions
    """
    
    def __init__(self):
        """Initialize content analyzer"""
        self.scene_categories = [
            'unboxing', 'review', 'tutorial', 'demonstration',
            'comparison', 'hands-on', 'showcase', 'features'
        ]
    
    def analyze_content(self, text_data: List[str], visual_features=None) -> Dict:
        """
        Analyze video content and generate description
        
        Args:
            text_data: List of text extracted from video
            visual_features: Optional visual features from CNN
            
        Returns:
            dict: Content analysis results
        """
        combined_text = ' '.join(text_data).lower()
        
        result = {
            'content_type': self._identify_content_type(combined_text),
            'main_topics': self._extract_main_topics(combined_text),
            'sentiment': self._basic_sentiment(combined_text),
            'description': None
        }
        
        # Generate description
        result['description'] = self._generate_description(result)
        
        return result
    
    def _identify_content_type(self, text: str) -> str:
        """Identify the type of content in the video"""
        for category in self.scene_categories:
            if category in text:
                return category
        
        # Fallback heuristics
        if any(word in text for word in ['new', 'first', 'look', 'impressions']):
            return 'review'
        elif any(word in text for word in ['vs', 'versus', 'compare']):
            return 'comparison'
        elif any(word in text for word in ['how', 'guide', 'tutorial']):
            return 'tutorial'
        else:
            return 'showcase'
    
    def _extract_main_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        topics = []
        
        topic_keywords = {
            'performance': ['fast', 'speed', 'performance', 'powerful'],
            'design': ['design', 'look', 'style', 'aesthetic', 'build'],
            'features': ['feature', 'capability', 'function'],
            'price': ['price', 'cost', 'expensive', 'affordable', 'value'],
            'comparison': ['vs', 'versus', 'compare', 'better', 'best']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _basic_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'best', 'love', 'perfect']
        negative_words = ['bad', 'worst', 'terrible', 'poor', 'disappointed', 'issue', 'problem']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_description(self, analysis: Dict) -> str:
        """Generate a description of the video content"""
        content_type = analysis['content_type']
        topics = analysis['main_topics']
        sentiment = analysis['sentiment']
        
        description = f"This appears to be a {content_type} video"
        
        if topics:
            description += f" focusing on {', '.join(topics[:3])}"
        
        description += f" with a {sentiment} tone."
        
        return description


if __name__ == "__main__":
    print("Keyword Extractor initialized")
    if EASYOCR_AVAILABLE:
        extractor = KeywordExtractor()
        print(f"OCR available with languages: {extractor.languages}")
    else:
        print("EasyOCR not available. Install with: pip install easyocr")
