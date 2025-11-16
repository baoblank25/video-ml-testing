"""
Configuration file for Video Analyzer
Customize settings here
"""

# Device settings
DEVICE = 'auto'  # 'auto', 'cuda', or 'cpu'

# Video processing settings
NUM_FRAMES = 12  # Number of frames to extract from each video
FRAME_EXTRACTION_METHOD = 'keyframe'  # 'keyframe' or 'uniform'

# CNN Model settings
CNN_MODEL = 'resnet50'  # 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3', 'vit_b_16'

# OCR settings
OCR_LANGUAGES = ['en']  # Languages for text extraction
OCR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for text extraction

# Output settings
SAVE_FRAMES = True  # Save extracted frames to disk
OUTPUT_DIR = 'results'  # Directory for output files

# Video URLs to analyze (customize this!)
VIDEO_URLS = [
    "https://www.youtube.com/shorts/MzIen6fSQwA",
    "https://www.youtube.com/shorts/9tMTeEMrpOM"
]

# Advanced settings
BATCH_SIZE = 8  # Batch size for CNN processing
USE_PRETRAINED = True  # Use pretrained CNN weights
VERBOSE = True  # Show detailed progress
