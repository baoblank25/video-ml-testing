"""
Ensemble CNN Model - Multi-Model Approach like Google Lens
Combines multiple CNN models for superior accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import Counter

from models.cnn_models import PretrainedCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleCNN:
    """
    Ensemble of multiple CNN models for improved accuracy
    Similar to Google Lens approach
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize ensemble with multiple models
        
        Args:
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device
        logger.info("Building Ensemble CNN Model (Google Lens approach)...")
        
        # Initialize multiple models
        self.models = {}
        
        # Model 1: ResNet50 - Good for general object recognition
        logger.info("Loading ResNet50...")
        self.models['resnet50'] = PretrainedCNN(
            model_name='resnet50',
            pretrained=True,
            device=device
        )
        
        # Model 2: EfficientNet B3 - Good for fine-grained details
        logger.info("Loading EfficientNet B3...")
        self.models['efficientnet_b3'] = PretrainedCNN(
            model_name='efficientnet_b3',
            pretrained=True,
            device=device
        )
        
        # Model 3: Vision Transformer - Good for global context
        logger.info("Loading Vision Transformer...")
        self.models['vit_b_16'] = PretrainedCNN(
            model_name='vit_b_16',
            pretrained=True,
            device=device
        )
        
        # Weights for each model (can be tuned)
        self.model_weights = {
            'resnet50': 0.35,
            'efficientnet_b3': 0.35,
            'vit_b_16': 0.30
        }
        
        logger.info(f"✓ Ensemble initialized with {len(self.models)} models")
    
    def extract_features(self, image) -> Dict[str, torch.Tensor]:
        """
        Extract features from all models in ensemble
        
        Args:
            image: Input image
            
        Returns:
            dict: Features from each model
        """
        features = {}
        
        for model_name, model in self.models.items():
            try:
                feat = model.extract_features(image)
                features[model_name] = feat
            except Exception as e:
                logger.warning(f"Error extracting features from {model_name}: {e}")
        
        return features
    
    def classify(self, image, return_probs=True) -> Dict:
        """
        Classify image using ensemble voting
        
        Args:
            image: Input image
            return_probs (bool): Return probabilities
            
        Returns:
            dict: Ensemble classification results
        """
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                pred = model.classify(image, return_probs=True)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Error in {model_name}: {e}")
        
        # Ensemble voting
        if predictions:
            ensemble_result = self._weighted_ensemble(predictions)
            return ensemble_result
        
        return None
    
    def _weighted_ensemble(self, predictions: Dict) -> Dict:
        """
        Combine predictions using weighted averaging
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            dict: Weighted ensemble result
        """
        # Weighted average of probabilities
        ensemble_probs = None
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            
            # Squeeze batch dimension if present
            if pred.dim() > 1:
                pred = pred.squeeze(0)
            
            if ensemble_probs is None:
                ensemble_probs = pred * weight
            else:
                ensemble_probs += pred * weight
            
            total_weight += weight
        
        if ensemble_probs is not None and total_weight > 0:
            ensemble_probs /= total_weight
        
        # Get top predictions
        top_k = 5
        probs, indices = torch.topk(ensemble_probs, top_k)
        
        return {
            'probabilities': probs,
            'indices': indices,
            'ensemble_probs': ensemble_probs,
            'individual_predictions': predictions
        }
    
    def batch_extract_features(self, images: List) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple images using ensemble
        
        Args:
            images: List of images
            
        Returns:
            dict: Aggregated features from all models
        """
        all_features = {model_name: [] for model_name in self.models.keys()}
        
        for image in images:
            features = self.extract_features(image)
            for model_name, feat in features.items():
                all_features[model_name].append(feat)
        
        # Stack features
        stacked_features = {}
        for model_name, feats in all_features.items():
            if feats:
                stacked_features[model_name] = torch.stack(feats)
        
        return stacked_features
    
    def get_consensus_prediction(self, images: List) -> Dict:
        """
        Get consensus prediction across multiple images
        
        Args:
            images: List of images from video frames
            
        Returns:
            dict: Consensus prediction with confidence
        """
        all_predictions = []
        
        for image in images:
            result = self.classify(image)
            if result:
                all_predictions.append(result)
        
        # Aggregate predictions across frames
        if not all_predictions:
            return None
        
        # Average probabilities across all frames
        avg_probs = torch.stack([p['ensemble_probs'] for p in all_predictions]).mean(dim=0)
        
        # Ensure avg_probs is 1D (remove any batch dimensions)
        while avg_probs.dim() > 1:
            avg_probs = avg_probs.squeeze(0)
        
        # Get top predictions
        top_k = 10
        probs, indices = torch.topk(avg_probs, top_k)
        
        # Convert to Python native types
        # probs and indices should be 1D tensors, use .tolist() for clean conversion
        probs_list = probs.tolist()
        indices_list = indices.tolist()
        
        return {
            'top_predictions': list(zip(indices_list, probs_list)),
            'confidence': probs_list[0] if len(probs_list) > 0 else 0.0,
            'num_frames_analyzed': len(all_predictions)
        }


class EnsembleProductClassifier:
    """
    Enhanced product classifier using ensemble model
    """
    
    def __init__(self, ensemble_model, device='cuda'):
        """
        Initialize with ensemble model
        
        Args:
            ensemble_model: EnsembleCNN instance
            device: Device to use
        """
        self.ensemble = ensemble_model
        self.device = device
        
        # Load ImageNet labels for better product identification
        self.imagenet_labels = self._load_full_imagenet_labels()
    
    def _load_full_imagenet_labels(self) -> Dict:
        """Load comprehensive ImageNet labels"""
        # Extended mapping for tech products
        labels = {
            # Laptops and computers
            609: 'laptop computer',
            620: 'computer mouse',
            722: 'notebook computer',
            782: 'screen, display',
            815: 'monitor, CRT screen',
            870: 'desktop computer',
            
            # Mobile devices
            487: 'cellular phone, mobile phone',
            648: 'iPod, smartphone',
            
            # Audio devices
            555: 'headphone, headset',
            831: 'speaker, loudspeaker',
            
            # Other tech
            510: 'remote control',
            532: 'digital watch',
            826: 'television, TV',
            
            # Common items
            504: 'coffee mug, cup',
            648: 'teacup, tea cup',
            960: 'wooden spoon',
            659: 'screen, monitor'
        }
        return labels
    
    def identify_product_from_frames(self, frames: List) -> Dict:
        """
        Identify product using ensemble across multiple frames
        
        Args:
            frames: List of video frames
            
        Returns:
            dict: Enhanced product identification
        """
        logger.info(f"Analyzing {len(frames)} frames with ensemble model...")
        
        # Get ensemble consensus
        consensus = self.ensemble.get_consensus_prediction(frames)
        
        if not consensus:
            return {'error': 'Could not analyze frames'}
        
        # Map predictions to product labels
        predictions = []
        for idx, prob in consensus['top_predictions']:
            # idx is already a Python int from .item(), no conversion needed
            if idx in self.imagenet_labels:
                predictions.append({
                    'label': self.imagenet_labels[idx],
                    'confidence': float(prob),
                    'category': self._categorize_label(self.imagenet_labels[idx])
                })
        
        result = {
            'ensemble_predictions': predictions,
            'primary_prediction': predictions[0] if predictions else None,
            'confidence': consensus['confidence'],
            'frames_analyzed': consensus['num_frames_analyzed'],
            'method': 'ensemble_cnn'
        }
        
        return result
    
    def _categorize_label(self, label: str) -> str:
        """Categorize product label"""
        label_lower = label.lower()
        
        if any(word in label_lower for word in ['laptop', 'notebook', 'computer']):
            return 'Laptops'
        elif any(word in label_lower for word in ['phone', 'mobile', 'cellular', 'ipod']):
            return 'Smartphones'
        elif 'mouse' in label_lower:
            return 'Computer Accessories'
        elif any(word in label_lower for word in ['headphone', 'headset', 'speaker']):
            return 'Audio Devices'
        elif any(word in label_lower for word in ['screen', 'monitor', 'display', 'tv']):
            return 'Displays'
        elif any(word in label_lower for word in ['cup', 'mug', 'tea']):
            return 'Beverages/Drinkware'
        else:
            return 'General Product'


if __name__ == "__main__":
    # Test ensemble
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing Ensemble CNN on {device}")
    
    ensemble = EnsembleCNN(device=device)
    print(f"✓ Ensemble loaded with {len(ensemble.models)} models")
    
    classifier = EnsembleProductClassifier(ensemble, device=device)
    print("✓ Ensemble classifier ready")
