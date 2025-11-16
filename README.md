# YouTube Video Analyzer - AI Product Identification System

An AI-powered system using Convolutional Neural Networks (CNNs) to analyze YouTube shorts and videos, identifying products, extracting keywords, and providing detailed content analysis.

## 🎯 Features

- **🤖 CNN-Based Visual Analysis**: Uses ResNet50 and other pretrained models for image classification
- **📹 Video Processing**: Downloads and extracts key frames from YouTube videos
- **🔍 Product Identification**: Identifies and categorizes products (laptops, phones, tablets, etc.)
- **📝 OCR & Keyword Extraction**: Extracts text and keywords from video frames
- **🏷️ Brand & Model Recognition**: Identifies specific brands and models (Dell XPS 15, MacBook Pro, etc.)
- **📊 Content Analysis**: Analyzes video type (review, unboxing, tutorial, etc.)
- **🎨 Multi-Frame Analysis**: Aggregates results across multiple frames for accuracy

## 📋 What It Does

For each YouTube video, the AI will provide:

1. **Product Identification**
   - Product name (e.g., "Dell XPS 15 Laptop")
   - Brand (e.g., "Dell")
   - Model (e.g., "XPS 15")
   - Category (e.g., "Laptops")
   - Confidence score

2. **Keywords**
   - Top keywords from the video
   - Brands mentioned
   - Technical specifications (RAM, storage, etc.)
   - Product types

3. **Content Analysis**
   - Video type (review, unboxing, comparison, etc.)
   - Main topics discussed
   - Sentiment analysis
   - Video description

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Install Dependencies

```powershell
# Install all required packages
.\.venv\Scripts\python.exe -m pip install torch torchvision opencv-python yt-dlp Pillow numpy pandas matplotlib scikit-learn easyocr transformers sentence-transformers requests tqdm imageio ffmpeg-python
```

Or using the requirements file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

```python
from analyze_videos import VideoAnalyzer

# Initialize the analyzer
analyzer = VideoAnalyzer(device='cuda', num_frames=10)

# Analyze a YouTube video
url = "https://www.youtube.com/shorts/YOUR_VIDEO_ID"
result = analyzer.analyze_video_url(url)

# Print the summary
analyzer.print_summary(result)
```

### Analyze Your Videos

```python
from analyze_videos import VideoAnalyzer

# Initialize
analyzer = VideoAnalyzer(device='cpu', num_frames=12)

# Analyze the two provided YouTube shorts
urls = [
    "https://www.youtube.com/shorts/MzIen6fSQwA",
    "https://www.youtube.com/shorts/9tMTeEMrpOM"
]

results = analyzer.analyze_multiple_videos(urls)

# Print results
for result in results:
    analyzer.print_summary(result)
    
# Save results to JSON
analyzer.save_results(results[0], "video_1_analysis.json")
```

### Example Output

```
==================================================================
VIDEO ANALYSIS SUMMARY
==================================================================

📹 Video Title: Dell XPS 15 Review 2024
⏱️  Duration: 60 seconds

🏷️  PRODUCT IDENTIFICATION:
   Product: Dell XPS 15
   Brand: Dell
   Model: XPS 15
   Category: Laptops
   Confidence: 87.5%

🔑 TOP KEYWORDS:
   • dell
   • xps
   • laptop
   • 15
   • screen
   • display
   • performance
   • intel
   • core
   • ssd

🏢 Brands Mentioned: Dell, Intel
⚙️  Specifications: 15 inch, 16gb ram, 512gb ssd, intel core, fhd

📊 CONTENT ANALYSIS:
   Type: review
   Topics: design, performance, features
   Sentiment: positive

📝 SUMMARY:
   This is a review video about the Dell XPS 15, which is categorized 
   as a Laptops. Key topics include: dell, xps, laptop, 15, screen.
==================================================================
```

## 🏗️ Project Structure

```
computer vision test/
├── models/
│   ├── cnn_models.py          # CNN architectures (ResNet, EfficientNet, etc.)
│   └── product_classifier.py  # Product classification system
├── utils/
│   ├── video_downloader.py    # YouTube video downloader
│   ├── frame_extractor.py     # Video frame extraction
│   └── keyword_extractor.py   # OCR and keyword extraction
├── data/                       # Downloaded videos
├── preprocessed/               # Extracted frames
├── analyze_videos.py           # Main application
├── example_usage.py            # Example scripts
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Components

### 1. CNN Models (`models/cnn_models.py`)

Implements multiple CNN architectures:
- **Custom ProductCNN**: Custom architecture for product classification
- **PretrainedCNN**: Wrapper for ResNet50, ResNet101, EfficientNet, ViT
- Features: Transfer learning, feature extraction, batch processing

### 2. Product Classifier (`models/product_classifier.py`)

- Classifies products into categories (Laptops, Smartphones, Tablets, etc.)
- Identifies specific brands and models
- Aggregates predictions across multiple frames
- Provides confidence scores

### 3. Video Downloader (`utils/video_downloader.py`)

- Downloads YouTube videos using yt-dlp
- Extracts video metadata (title, description, tags)
- Handles shorts and regular videos
- Supports batch downloading

### 4. Frame Extractor (`utils/frame_extractor.py`)

- Extracts key frames from videos
- Two methods: uniform sampling and keyframe detection
- Preprocesses frames for CNN input
- Detects scene changes

### 5. Keyword Extractor (`utils/keyword_extractor.py`)

- OCR using EasyOCR
- Extracts text from video frames
- Categorizes keywords (brands, specs, models)
- Content analysis and sentiment detection

## 🎓 How It Works

1. **Download**: Downloads the YouTube video using yt-dlp
2. **Extract Frames**: Extracts 10-15 key frames from the video
3. **Visual Analysis**: Runs frames through CNN for product classification
4. **Text Extraction**: Uses OCR to extract text from frames
5. **Keyword Analysis**: Identifies brands, models, and specifications
6. **Aggregation**: Combines results from all frames
7. **Report Generation**: Creates comprehensive analysis report

## 💡 Advanced Usage

### Use Different CNN Models

```python
from models.cnn_models import PretrainedCNN

# Use EfficientNet instead of ResNet
cnn = PretrainedCNN(model_name='efficientnet_b3', device='cuda')
```

### Extract More Frames

```python
# Extract 20 frames for more detailed analysis
analyzer = VideoAnalyzer(device='cuda', num_frames=20)
```

### Custom Frame Extraction

```python
from utils.frame_extractor import FrameExtractor

extractor = FrameExtractor()
frames = extractor.extract_frames(
    'video.mp4',
    num_frames=15,
    method='keyframe'  # or 'uniform'
)
```

### Analyze Local Videos

```python
from utils.frame_extractor import FrameExtractor
from models.cnn_models import PretrainedCNN
from models.product_classifier import ProductClassifier

# Extract frames from local video
extractor = FrameExtractor()
frames = extractor.extract_frames('my_video.mp4', num_frames=10)

# Analyze with CNN
cnn = PretrainedCNN(model_name='resnet50')
classifier = ProductClassifier(cnn)
results = classifier.identify_product_from_frames(frames)

print(results)
```

## 🎯 Supported Products

The system can identify:

- **Laptops**: Dell XPS, MacBook, HP Spectre, Lenovo ThinkPad, ASUS ROG, etc.
- **Smartphones**: iPhone, Samsung Galaxy, Google Pixel, OnePlus, etc.
- **Tablets**: iPad, Samsung Tab, Surface Pro, Kindle
- **Headphones**: AirPods, Sony WH, Bose, Beats
- **Smartwatches**: Apple Watch, Samsung Galaxy Watch, Fitbit
- **Gaming Consoles**: PlayStation, Xbox, Nintendo Switch
- **And many more tech products**

## 🔍 Customization

### Add New Product Categories

Edit `models/product_classifier.py`:

```python
self.product_categories = {
    'YourCategory': {
        'Subcategory': ['Product1', 'Product2', ...]
    }
}
```

### Add New Keywords

Edit `utils/keyword_extractor.py`:

```python
self.tech_keywords = {
    'your_category': ['keyword1', 'keyword2', ...]
}
```

## ⚡ Performance Tips

1. **Use GPU**: Set `device='cuda'` for 10x faster processing
2. **Adjust Frame Count**: Fewer frames = faster processing (but less accuracy)
3. **Use Smaller Models**: `efficientnet_b0` is faster than `resnet101`
4. **Batch Processing**: Analyze multiple videos in one run

## 🐛 Troubleshooting

### EasyOCR Installation Issues

```powershell
# Install EasyOCR separately
.\.venv\Scripts\python.exe -m pip install easyocr
```

### CUDA Out of Memory

```python
# Use CPU instead
analyzer = VideoAnalyzer(device='cpu', num_frames=10)
```

### Video Download Fails

- Check internet connection
- Verify YouTube URL is accessible
- Try updating yt-dlp: `pip install --upgrade yt-dlp`

## 📊 Output Format

Results are saved in JSON format:

```json
{
  "video_title": "Product Review",
  "product_identification": {
    "product_name": "Dell XPS 15",
    "brand": "Dell",
    "model": "XPS 15",
    "category": "Laptops",
    "confidence": 0.875
  },
  "keywords": {
    "top_keywords": ["dell", "xps", "laptop", ...],
    "brands_mentioned": ["Dell", "Intel"],
    "specifications": ["16gb", "512gb ssd", ...]
  },
  "content_analysis": {
    "content_type": "review",
    "main_topics": ["performance", "design"],
    "sentiment": "positive"
  },
  "summary": "This is a review video about the Dell XPS 15..."
}
```

## 🚀 Running the Full Analysis

To analyze your YouTube shorts:

```powershell
# Run the main analyzer
.\.venv\Scripts\python.exe analyze_videos.py
```

Or run the example:

```powershell
# Run the example script
.\.venv\Scripts\python.exe example_usage.py
```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to customize and extend the system for your needs!

## 📧 Support

For issues or questions, check the troubleshooting section or review the code comments.

---

**Built with ❤️ using PyTorch, OpenCV, and EasyOCR**
