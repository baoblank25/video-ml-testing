"""
Convolutional Neural Network Models
CNN architectures for video frame analysis and feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductCNN(nn.Module):
    """
    Custom CNN for product classification and feature extraction
    """
    
    def __init__(self, num_classes=100):
        """
        Initialize the CNN
        
        Args:
            num_classes (int): Number of product categories
        """
        super(ProductCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def extract_features(self, x):
        """Extract feature vectors from input images"""
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Extract features before final classification
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        return features


class PretrainedCNN:
    """
    Wrapper for pretrained CNN models (ResNet, EfficientNet, etc.)
    """
    
    def __init__(self, model_name='resnet50', pretrained=True, device='cuda'):
        """
        Initialize pretrained model
        
        Args:
            model_name (str): Name of the model (resnet50, resnet101, efficientnet_b0, etc.)
            pretrained (bool): Use pretrained weights
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif model_name == 'efficientnet_b3':
            self.model = models.efficientnet_b3(pretrained=pretrained)
            self.feature_dim = 1536
        elif model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=pretrained)
            self.feature_dim = 768
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Loaded {model_name} on {self.device}")
    
    def extract_features(self, image):
        """
        Extract feature vectors from an image
        
        Args:
            image (numpy.ndarray): Input image (H, W, C)
            
        Returns:
            torch.Tensor: Feature vector
        """
        # Preprocess image
        if isinstance(image, torch.Tensor):
            img_tensor = image
        else:
            img_tensor = self.preprocess(image)
        
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            if 'resnet' in self.model_name:
                # Remove final FC layer
                features = self.model.avgpool(self.model.layer4(
                    self.model.layer3(self.model.layer2(
                        self.model.layer1(self.model.maxpool(
                            self.model.relu(self.model.bn1(
                                self.model.conv1(img_tensor)
                            ))
                        ))
                    ))
                ))
                features = torch.flatten(features, 1)
            elif 'efficientnet' in self.model_name:
                features = self.model.features(img_tensor)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
            elif 'vit' in self.model_name:
                features = self.model._process_input(img_tensor)
                features = self.model.encoder(features)
                features = features[:, 0]  # Class token
            else:
                features = self.model(img_tensor)
        
        return features
    
    def classify(self, image, return_probs=True):
        """
        Classify an image
        
        Args:
            image (numpy.ndarray): Input image
            return_probs (bool): Return probabilities instead of logits
            
        Returns:
            torch.Tensor: Classification scores
        """
        # Preprocess image
        if isinstance(image, torch.Tensor):
            img_tensor = image
        else:
            img_tensor = self.preprocess(image)
        
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model(img_tensor)
            
            if return_probs:
                output = F.softmax(output, dim=1)
        
        return output
    
    def batch_extract_features(self, images):
        """
        Extract features from multiple images
        
        Args:
            images (list): List of images
            
        Returns:
            torch.Tensor: Batch of feature vectors
        """
        # Preprocess all images
        img_tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(img_tensors).to(self.device)
        
        # Extract features
        with torch.no_grad():
            if 'resnet' in self.model_name:
                features = self.model.avgpool(self.model.layer4(
                    self.model.layer3(self.model.layer2(
                        self.model.layer1(self.model.maxpool(
                            self.model.relu(self.model.bn1(
                                self.model.conv1(batch)
                            ))
                        ))
                    ))
                ))
                features = torch.flatten(features, 1)
            elif 'efficientnet' in self.model_name:
                features = self.model.features(batch)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
            else:
                features = self.model(batch)
        
        return features


if __name__ == "__main__":
    # Test the models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test custom CNN
    model = ProductCNN(num_classes=100)
    print(f"Custom CNN created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test pretrained model
    pretrained = PretrainedCNN(model_name='resnet50', device=device)
    print(f"Pretrained model loaded: {pretrained.model_name}")
