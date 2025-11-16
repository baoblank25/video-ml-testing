"""
Smart Product Detector
Combines CNN, OCR, and video metadata for accurate product identification
"""

import re
import logging
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class SmartProductDetector:
    """
    Intelligent product detection using multiple data sources
    """
    
    def __init__(self):
        """Initialize smart detector with product databases"""
        
        # Laptop brands and models
        self.laptop_patterns = {
            'Dell': {
                'models': ['XPS 13', 'XPS 15', 'XPS 17', 'Inspiron', 'Latitude', 'Alienware', 'Precision'],
                'regex': r'(?i)(dell\s+)?xps\s*(\d+|15-?\d+)',
            },
            'Apple': {
                'models': ['MacBook Pro', 'MacBook Air', 'MacBook'],
                'regex': r'(?i)macbook\s*(pro|air)?',
            },
            'HP': {
                'models': ['Spectre', 'Envy', 'Pavilion', 'EliteBook', 'Omen'],
                'regex': r'(?i)hp\s+(spectre|envy|pavilion|elitebook|omen)',
            },
            'Lenovo': {
                'models': ['ThinkPad', 'IdeaPad', 'Legion', 'Yoga'],
                'regex': r'(?i)(thinkpad|ideapad|legion|yoga)',
            },
            'ASUS': {
                'models': ['ROG', 'ZenBook', 'VivoBook', 'TUF'],
                'regex': r'(?i)asus\s+(rog|zenbook|vivobook|tuf)',
            },
        }
        
        # Beverage brands
        self.beverage_patterns = {
            'Tenzo Tea': {
                'products': ['Ceremonial Matcha', 'Matcha Powder'],
                'regex': r'(?i)tenzo\s*tea',
                'category': 'Matcha'
            },
            'Ippodo': {
                'products': ['Matcha', 'Green Tea'],
                'regex': r'(?i)ippodo',
                'category': 'Matcha'
            },
            'Encha': {
                'products': ['Organic Matcha'],
                'regex': r'(?i)encha',
                'category': 'Matcha'
            },
            'Jade Leaf': {
                'products': ['Organic Matcha'],
                'regex': r'(?i)jade\s*leaf',
                'category': 'Matcha'
            },
        }
        
        # Product category keywords
        self.category_keywords = {
            'laptop': ['laptop', 'notebook', 'computer', 'pc', 'repair', 'motherboard', 'bios'],
            'matcha': ['matcha', 'green tea', 'latte', 'tea', 'ceremonial', 'powder'],
            'phone': ['phone', 'smartphone', 'iphone', 'galaxy', 'pixel'],
            'tablet': ['tablet', 'ipad', 'tab'],
            'keyboard': ['keyboard', 'mechanical keyboard', 'switches', 'keycaps', 'keyswitches', 'typing'],
        }
    
    def detect_product(self, 
                      video_info: Dict,
                      ocr_keywords: Dict,
                      cnn_prediction: Dict) -> Dict:
        """
        Intelligently detect product using all available data
        
        Args:
            video_info: Video metadata (title, description, tags)
            ocr_keywords: Extracted keywords from OCR
            cnn_prediction: CNN model predictions
            
        Returns:
            dict: Enhanced product identification
        """
        logger.info("🔍 Running smart product detection...")
        
        result = {
            'product_name': None,
            'brand': None,
            'model': None,
            'category': None,
            'confidence': 0.0,
            'detection_sources': [],
            'raw_cnn': cnn_prediction,
        }
        
        # Combine all text sources
        all_text = self._combine_text_sources(video_info, ocr_keywords)
        
        # Step 1: Detect category from context
        category = self._detect_category(all_text, video_info)
        result['category'] = category
        logger.info(f"  📁 Detected category: {category}")
        
        # Step 2: Universal brand/product detection - AUTOMATIC!
        # Try all detection methods automatically, no need to add new code
        brand_info = self._auto_detect_product(all_text, video_info, ocr_keywords, category)
        if brand_info:
            result.update(brand_info)
            result['detection_sources'].append(brand_info.get('source', 'Auto-Detection'))
            logger.info(f"  ✅ Auto-detected: {brand_info}")
        
        # Step 3: Calculate confidence based on detection sources
        result['confidence'] = self._calculate_confidence(result, cnn_prediction)
        
        # Step 4: Create full product name
        if result['brand'] and result['model']:
            result['product_name'] = f"{result['brand']} {result['model']}"
        elif result['brand']:
            result['product_name'] = result['brand']
        elif not result['product_name'] and cnn_prediction.get('product_name'):
            result['product_name'] = cnn_prediction['product_name']
        
        return result
    
    def _auto_detect_product(self, text: str, video_info: Dict, ocr_keywords: Dict, category: str) -> Optional[Dict]:
        """
        Universal automatic product detection - works for ANY product!
        No need to add new methods for new product types.
        """
        # Try specific detectors first (they have more detailed patterns)
        if 'laptop' in category.lower() or 'computer' in category.lower():
            laptop_result = self._detect_laptop(text, ocr_keywords)
            if laptop_result and laptop_result.get('brand'):
                laptop_result['source'] = 'Laptop Detection'
                return laptop_result
        
        if 'matcha' in category.lower() or 'beverage' in category.lower() or 'tea' in category.lower():
            beverage_result = self._detect_beverage(text)
            if beverage_result and beverage_result.get('brand'):
                beverage_result['source'] = 'Beverage Detection'
                return beverage_result
        
        if 'keyboard' in category.lower() or 'switch' in category.lower() or 'peripheral' in category.lower():
            keyboard_result = self._detect_keyboard(text, video_info)
            if keyboard_result and keyboard_result.get('brand'):
                keyboard_result['source'] = 'Keyboard Detection'
                return keyboard_result
        
        # Universal fallback: Extract brand/product from title and metadata
        # This works for ANY product without needing special code!
        result = {
            'product_name': None,
            'brand': None,
            'model': None,
            'category': category,
            'source': 'Title + Metadata'
        }
        
        title = video_info.get('title', '').strip()
        description = video_info.get('description', '').strip()
        tags = video_info.get('tags', [])
        
        # Extract capitalized words as potential brand names
        title_words = title.split()
        potential_brands = [w for w in title_words if w and w[0].isupper() and len(w) > 2]
        
        # Common brand indicators
        brand_indicators = ['brand', 'official', 'by', 'from']
        for tag in tags:
            tag_lower = tag.lower()
            if any(indicator in tag_lower for indicator in brand_indicators):
                potential_brands.append(tag)
        
        if potential_brands:
            result['brand'] = potential_brands[0]
        
        # Use title as product name if no specific detection
        if not result['product_name'] and title:
            # Clean title (remove common YouTube words)
            clean_title = re.sub(r'(?i)(review|unboxing|trying|testing|shorts|#\w+)', '', title).strip()
            result['product_name'] = clean_title[:50]  # Limit length
        
        return result if result['brand'] or result['product_name'] else None
    
    def _combine_text_sources(self, video_info: Dict, ocr_keywords: Dict) -> str:
        """Combine all text from video metadata and OCR"""
        text_parts = []
        
        if video_info:
            text_parts.append(video_info.get('title', ''))
            text_parts.append(video_info.get('description', ''))
            text_parts.extend(video_info.get('tags', []))
        
        if ocr_keywords:
            text_parts.extend(ocr_keywords.get('all_text', []))
            text_parts.extend(ocr_keywords.get('high_confidence_text', []))
        
        return ' '.join(text_parts).lower()
    
    def _detect_category(self, text: str, video_info: Dict) -> str:
        """Detect product category from text"""
        text_lower = text.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return 'General Product'
        
        # Get category with highest score
        best_category = max(category_scores, key=category_scores.get)
        
        # Map to standard categories
        category_map = {
            'laptop': 'Laptops',
            'matcha': 'Matcha/Beverages',
            'phone': 'Smartphones',
            'tablet': 'Tablets',
            'keyboard': 'Keyboards/Peripherals',
        }
        
        return category_map.get(best_category, 'Electronics')
    
    def _detect_laptop(self, text: str, ocr_keywords: Dict) -> Optional[Dict]:
        """Detect laptop brand and model"""
        text_lower = text.lower()
        
        # Check each laptop brand
        for brand, patterns in self.laptop_patterns.items():
            # Try regex match first
            if 'regex' in patterns:
                match = re.search(patterns['regex'], text_lower)
                if match:
                    model = None
                    
                    # Extract specific model number
                    if 'xps' in text_lower:
                        # Look for XPS model number
                        xps_match = re.search(r'xps\s*(\d+)', text_lower)
                        if xps_match:
                            model = f"XPS {xps_match.group(1)}"
                        else:
                            model = "XPS"
                    
                    # Check OCR for model numbers
                    if ocr_keywords and 'model_identifiers' in ocr_keywords:
                        for item in ocr_keywords['model_identifiers']:
                            keyword = item['keyword'] if isinstance(item, dict) else item
                            if 'xps' in keyword.lower():
                                model = keyword.upper()
                    
                    return {
                        'brand': brand,
                        'model': model or brand,
                        'confidence': 0.85,
                    }
            
            # Check for model names
            for model in patterns['models']:
                if model.lower() in text_lower:
                    return {
                        'brand': brand,
                        'model': model,
                        'confidence': 0.80,
                    }
        
        return None
    
    def _detect_beverage(self, text: str) -> Optional[Dict]:
        """Detect beverage brand"""
        text_lower = text.lower()
        
        for brand, patterns in self.beverage_patterns.items():
            # Try regex match
            if 'regex' in patterns:
                match = re.search(patterns['regex'], text_lower)
                if match:
                    return {
                        'brand': brand,
                        'model': patterns.get('category', 'Matcha'),
                        'product_category': patterns.get('category'),
                        'confidence': 0.90,
                    }
            
            # Check for product names
            if 'products' in patterns:
                for product in patterns['products']:
                    if product.lower() in text_lower:
                        return {
                            'brand': brand,
                            'model': product,
                            'product_category': patterns.get('category'),
                            'confidence': 0.85,
                        }
        
        # Generic matcha detection
        if 'matcha' in text_lower:
            return {
                'brand': 'Matcha Brand (Unknown)',
                'model': 'Matcha Powder',
                'product_category': 'Matcha',
                'confidence': 0.50,
            }
        
        return None
    
    def _detect_keyboard(self, text: str, video_info: Dict) -> Optional[Dict]:
        """Detect keyboard switches and peripherals from video title"""
        text_lower = text.lower()
        
        # Common switch brands and types
        switch_brands = {
            'cherry': ['cherry mx', 'cherry', 'mx red', 'mx blue', 'mx brown', 'mx black'],
            'gateron': ['gateron', 'gateron red', 'gateron blue', 'gateron yellow'],
            'kailh': ['kailh', 'box switches', 'kailh box'],
            'outemu': ['outemu'],
            'holy panda': ['holy panda'],
            'zealios': ['zealios', 'zealpc'],
            'glorious': ['glorious panda'],
            'akko': ['akko'],
            'ttc': ['ttc'],
        }
        
        # Check for switch brands
        for brand, variations in switch_brands.items():
            for variation in variations:
                if variation in text_lower:
                    return {
                        'brand': brand.title(),
                        'model': 'Mechanical Switches',
                        'product_category': 'Keyboard Switches',
                        'confidence': 0.85,
                    }
        
        # Generic switch detection from title
        title = video_info.get('title', '')
        if 'switches' in text_lower or 'switch' in text_lower:
            # Extract switch name from title
            import re
            # Look for pattern like "Diamond Switches", "Red Switches", etc.
            match = re.search(r'(\w+)\s+switch', title, re.IGNORECASE)
            if match:
                switch_name = match.group(1).title()
                return {
                    'brand': f'{switch_name} Switches',
                    'model': 'Mechanical Keyboard Switches',
                    'product_category': 'Keyboard Switches',
                    'confidence': 0.75,
                }
        
        # Generic keyboard detection
        if 'keyboard' in text_lower or 'keycaps' in text_lower:
            return {
                'brand': 'Mechanical Keyboard',
                'model': 'Keyboard',
                'product_category': 'Keyboards',
                'confidence': 0.70,
            }
        
        return None
    
    def _calculate_confidence(self, result: Dict, cnn_prediction: Dict) -> float:
        """Calculate overall confidence score"""
        confidence = 0.0
        
        # Base confidence from detection
        if result.get('brand') and result.get('model'):
            confidence = 0.80  # High confidence from multiple sources
        elif result.get('brand'):
            confidence = 0.65  # Medium confidence
        
        # Boost from CNN if categories match
        if cnn_prediction and cnn_prediction.get('category'):
            cnn_category = cnn_prediction['category']
            detected_category = result.get('category', '')
            
            # Partial matches
            if cnn_category in detected_category or detected_category in cnn_category:
                confidence += 0.10
        
        # Boost from multiple detection sources
        if len(result.get('detection_sources', [])) > 1:
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 100%


def enhance_product_detection(video_info: Dict,
                              ocr_keywords: Dict,
                              cnn_results: Dict) -> Dict:
    """
    Main function to enhance product detection
    
    Args:
        video_info: Video metadata
        ocr_keywords: OCR extracted keywords
        cnn_results: CNN predictions
        
    Returns:
        dict: Enhanced product information
    """
    detector = SmartProductDetector()
    return detector.detect_product(video_info, ocr_keywords, cnn_results)
