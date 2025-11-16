"""
Simple Example: Analyze YouTube Videos
Quick start guide for using the Video Analyzer
"""

from analyze_videos import VideoAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def analyze_single_video():
    """Example: Analyze a single YouTube video"""
    
    print("\n=== Analyzing Single Video ===\n")
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(device='cpu', num_frames=10)
    
    # Your YouTube video URL
    url = "https://www.youtube.com/shorts/MzIen6fSQwA"
    
    # Analyze
    result = analyzer.analyze_video_url(url, video_name="my_video")
    
    # Print results
    analyzer.print_summary(result)
    
    # Access specific information
    print("\n=== Detailed Product Info ===")
    product = result['product_identification']
    print(f"Product: {product['product_name']}")
    print(f"Brand: {product['brand']}")
    print(f"Category: {product['category']}")
    
    # Save results
    analyzer.save_results(result, "my_video_analysis.json")


def analyze_multiple_videos():
    """Example: Analyze multiple YouTube videos"""
    
    print("\n=== Analyzing Multiple Videos ===\n")
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(device='cpu', num_frames=10)
    
    # List of video URLs
    urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM"
    ]
    
    # Analyze all videos
    results = analyzer.analyze_multiple_videos(urls)
    
    # Print summaries
    for result in results:
        analyzer.print_summary(result)
    
    # Compare products
    print("\n=== Product Comparison ===")
    for i, result in enumerate(results):
        product = result['product_identification']
        print(f"Video {i+1}: {product['product_name']} ({product['category']})")


def custom_analysis():
    """Example: Custom analysis with specific components"""
    
    from utils.video_downloader import VideoDownloader
    from utils.frame_extractor import FrameExtractor
    from utils.keyword_extractor import KeywordExtractor
    
    print("\n=== Custom Analysis ===\n")
    
    # Download video
    downloader = VideoDownloader()
    video_path = downloader.download_video(
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        filename="test_video"
    )
    
    # Extract frames
    extractor = FrameExtractor()
    frames = extractor.extract_frames(video_path, num_frames=5)
    print(f"Extracted {len(frames)} frames")
    
    # Extract keywords
    keyword_extractor = KeywordExtractor()
    keywords = keyword_extractor.extract_keywords_from_frames(frames)
    print(f"Found keywords: {list(keywords['keyword_counts'].keys())[:10]}")


if __name__ == "__main__":
    # Run the example you want:
    
    # Option 1: Analyze single video
    analyze_single_video()
    
    # Option 2: Analyze multiple videos
    # analyze_multiple_videos()
    
    # Option 3: Custom analysis
    # custom_analysis()
