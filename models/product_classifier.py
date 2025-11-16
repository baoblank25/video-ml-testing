"""
Product Classification and Identification System
Identifies and categorizes products from video frames
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductClassifier:
    """
    Classify and identify products from images
    """
    
    def __init__(self, cnn_model, device='cuda'):
        """
        Initialize the product classifier
        
        Args:
            cnn_model: CNN model for feature extraction
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model = cnn_model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Product categories and their subcategories
        self.product_categories = {
            'Electronics': {
                'Laptops': ['Dell XPS', 'MacBook', 'HP Spectre', 'Lenovo ThinkPad', 
                           'ASUS ROG', 'Surface Laptop', 'Razer Blade'],
                'Smartphones': ['iPhone', 'Samsung Galaxy', 'Google Pixel', 
                               'OnePlus', 'Xiaomi', 'Huawei'],
                'Tablets': ['iPad', 'Samsung Tab', 'Surface Pro', 'Kindle'],
                'Headphones': ['AirPods', 'Sony WH', 'Bose', 'Beats'],
                'Cameras': ['Canon', 'Nikon', 'Sony Alpha', 'GoPro'],
                'Gaming': ['PlayStation', 'Xbox', 'Nintendo Switch', 'Steam Deck']
            },
            'Accessories': {
                'Mouse': ['Logitech MX', 'Razer', 'Apple Magic Mouse'],
                'Keyboard': ['Mechanical', 'Wireless', 'Gaming'],
                'Charger': ['USB-C', 'Wireless', 'Power Bank'],
                'Case': ['Phone Case', 'Laptop Bag', 'Tablet Cover']
            },
            'Wearables': {
                'Smartwatch': ['Apple Watch', 'Samsung Galaxy Watch', 'Fitbit'],
                'Earbuds': ['AirPods', 'Galaxy Buds', 'Pixel Buds']
            }
        }
        
        # ImageNet class mapping for general object detection
        # (Simplified - in practice, load full ImageNet labels)
        self.imagenet_labels = self._load_imagenet_labels()
        
    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        # This is a simplified version. In production, load from file
        labels = {}
        # Common tech products in ImageNet
        labels.update({
            609: 'laptop, laptop computer',
            620: 'mouse, computer mouse',
            648: 'mobile phone, cellular phone',
            722: 'notebook, notebook computer',
            815: 'screen, CRT screen',
            782: 'speaker, loudspeaker',
            770: 'remote control',
            826: 'iPod'
        })
        return labels
    
    def classify_product(self, image) -> Dict:
        """
        Classify a product in an image
        
        Args:
            image: Input image (numpy array or tensor)
            
        Returns:
            dict: Classification results with product info
        """
        try:
            # Extract features
            features = self.model.extract_features(image)
            
            # Get classification scores
            scores = self.model.classify(image, return_probs=True)
            
            # Get top predictions
            top_k = 5
            probs, indices = torch.topk(scores, top_k)
            
            results = {
                'predictions': [],
                'main_category': None,
                'subcategory': None,
                'product_name': None,
                'confidence': 0.0
            }
            
            # Map to product categories
            for prob, idx in zip(probs[0], indices[0]):
                idx_val = idx.item()
                prob_val = prob.item()
                
                if idx_val in self.imagenet_labels:
                    label = self.imagenet_labels[idx_val]
                    results['predictions'].append({
                        'label': label,
                        'confidence': prob_val
                    })
            
            # Determine main category
            if results['predictions']:
                top_pred = results['predictions'][0]
                results['product_name'] = top_pred['label']
                results['confidence'] = top_pred['confidence']
                results['main_category'] = self._determine_category(top_pred['label'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying product: {str(e)}")
            return {'error': str(e)}
    
    def _determine_category(self, label):
        """Determine product category from label"""
        label_lower = label.lower()
        
        # Check against known categories
        if any(word in label_lower for word in ['laptop', 'notebook', 'computer']):
            return 'Laptops'
        elif any(word in label_lower for word in ['phone', 'mobile', 'cellular']):
            return 'Smartphones'
        elif 'tablet' in label_lower or 'ipad' in label_lower:
            return 'Tablets'
        elif any(word in label_lower for word in ['headphone', 'earbud', 'airpod']):
            return 'Headphones'
        elif 'watch' in label_lower:
            return 'Smartwatch'
        elif 'mouse' in label_lower:
            return 'Mouse'
        elif 'keyboard' in label_lower:
            return 'Keyboard'
        else:
            return 'Electronics'
    
    def identify_product_from_frames(self, frames: List) -> Dict:
        """
        Identify product across multiple video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            dict: Aggregated product identification results
        """
        all_predictions = []
        categories = []
        product_names = []
        
        # Analyze each frame
        for i, frame in enumerate(frames):
            result = self.classify_product(frame)
            
            if 'predictions' in result and result['predictions']:
                all_predictions.append(result)
                
                if result['main_category']:
                    categories.append(result['main_category'])
                
                if result['product_name']:
                    product_names.append(result['product_name'])
        
        # Aggregate results
        aggregated = {
            'most_common_category': None,
            'most_common_product': None,
            'confidence': 0.0,
            'frame_results': all_predictions,
            'summary': None
        }
        
        if categories:
            category_counts = Counter(categories)
            aggregated['most_common_category'] = category_counts.most_common(1)[0][0]
        
        if product_names:
            product_counts = Counter(product_names)
            most_common = product_counts.most_common(1)[0]
            aggregated['most_common_product'] = most_common[0]
            
            # Calculate average confidence
            confidences = [r['confidence'] for r in all_predictions if 'confidence' in r]
            if confidences:
                aggregated['confidence'] = np.mean(confidences)
        
        # Generate summary
        if aggregated['most_common_product'] and aggregated['most_common_category']:
            aggregated['summary'] = (
                f"Identified as {aggregated['most_common_product']} "
                f"in category {aggregated['most_common_category']} "
                f"with {aggregated['confidence']:.2%} confidence"
            )
        
        return aggregated
    
    def get_detailed_product_info(self, product_name: str) -> Dict:
        """
        Get detailed information about a specific product
        
        Args:
            product_name: Name of the product
            
        Returns:
            dict: Detailed product information
        """
        info = {
            'name': product_name,
            'category': None,
            'subcategory': None,
            'brands': [],
            'typical_use': None
        }
        
        product_lower = product_name.lower()
        
        # Match against known products
        for main_cat, subcats in self.product_categories.items():
            for subcat, products in subcats.items():
                for product in products:
                    if product.lower() in product_lower or product_lower in product.lower():
                        info['category'] = main_cat
                        info['subcategory'] = subcat
                        info['brands'].append(product)
        
        # Determine typical use
        if 'laptop' in product_lower:
            info['typical_use'] = 'Computing, work, gaming, content creation'
        elif 'phone' in product_lower:
            info['typical_use'] = 'Communication, mobile computing, photography'
        elif 'watch' in product_lower:
            info['typical_use'] = 'Fitness tracking, notifications, health monitoring'
        elif 'headphone' in product_lower or 'earbud' in product_lower:
            info['typical_use'] = 'Audio playback, music, calls, noise cancellation'
        
        return info


class ProductBrandIdentifier:
    """
    Identify specific product brands and models
    """
    
    def __init__(self):
        """Initialize brand identifier with known patterns"""
        self.brand_patterns = {
            'Dell': ['XPS', 'Inspiron', 'Latitude', 'Alienware', 'Precision'],
            'Apple': ['MacBook', 'iPhone', 'iPad', 'AirPods', 'Watch', 'iMac'],
            'Samsung': ['Galaxy', 'Note', 'Tab', 'Buds', 'Watch'],
            'HP': ['Spectre', 'Envy', 'Pavilion', 'EliteBook', 'Omen'],
            'Lenovo': ['ThinkPad', 'IdeaPad', 'Legion', 'Yoga'],
            'ASUS': ['ROG', 'ZenBook', 'VivoBook', 'TUF'],
            'Microsoft': ['Surface', 'Xbox'],
            'Sony': ['PlayStation', 'Alpha', 'WH-', 'Xperia'],
            'Google': ['Pixel', 'Nest', 'Chromecast'],
            'Nintendo': ['Switch', 'Joy-Con']
        }
    
    def identify_brand_and_model(self, text: str) -> Dict:
        """
        Identify brand and model from text
        
        Args:
            text: Text containing potential product information
            
        Returns:
            dict: Brand and model information
        """
        text_lower = text.lower()
        
        result = {
            'brand': None,
            'model': None,
            'full_name': None
        }
        
        # Check for brand patterns
        for brand, models in self.brand_patterns.items():
            if brand.lower() in text_lower:
                result['brand'] = brand
                
                # Check for specific models
                for model in models:
                    if model.lower() in text_lower:
                        result['model'] = model
                        result['full_name'] = f"{brand} {model}"
                        return result
                
                result['full_name'] = brand
                return result
        
        return result


if __name__ == "__main__":
    print("Product Classification System initialized")
    print("Categories:", list(ProductClassifier(None).product_categories.keys()))
