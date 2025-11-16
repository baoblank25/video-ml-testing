"""
YouTube Video Analyzer - Main Application
Complete AI system for analyzing YouTube shorts and extracting product information
"""

import os
import sys
from pathlib import Path
import logging
import json
from typing import List, Dict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.video_downloader import VideoDownloader
from utils.frame_extractor import FrameExtractor
from utils.keyword_extractor import KeywordExtractor, ContentAnalyzer
from models.cnn_models import PretrainedCNN
from models.product_classifier import ProductClassifier, ProductBrandIdentifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    Complete video analysis system combining all components
    """
    
    def __init__(self, device='cuda', num_frames=15):
        """
        Initialize the video analyzer
        
        Args:
            device (str): Device to use ('cuda' or 'cpu')
            num_frames (int): Number of frames to extract from each video
        """
        logger.info("Initializing Video Analyzer...")
        
        self.device = device
        self.num_frames = num_frames
        
        # Initialize components
        logger.info("Loading video downloader...")
        self.downloader = VideoDownloader(output_dir='data')
        
        logger.info("Loading frame extractor...")
        self.frame_extractor = FrameExtractor(output_dir='preprocessed')
        
        logger.info("Loading CNN model...")
        self.cnn_model = PretrainedCNN(
            model_name='resnet50',
            pretrained=True,
            device=device
        )
        
        logger.info("Loading product classifier...")
        self.product_classifier = ProductClassifier(
            cnn_model=self.cnn_model,
            device=device
        )
        
        logger.info("Loading keyword extractor...")
        self.keyword_extractor = KeywordExtractor(languages=['en'])
        
        logger.info("Loading content analyzer...")
        self.content_analyzer = ContentAnalyzer()
        
        logger.info("Loading brand identifier...")
        self.brand_identifier = ProductBrandIdentifier()
        
        logger.info("Video Analyzer initialized successfully!")
    
    def analyze_video_url(self, url: str, video_name: str = None) -> Dict:
        """
        Analyze a YouTube video from URL
        
        Args:
            url (str): YouTube video URL
            video_name (str, optional): Custom name for the video
            
        Returns:
            dict: Complete analysis results
        """
        logger.info(f"Starting analysis of video: {url}")
        
        try:
            # Step 1: Download video
            logger.info("Step 1: Downloading video...")
            video_path = self.downloader.download_video(url, filename=video_name)
            
            # Get video metadata
            video_info = self.downloader.get_video_info(url)
            
            # Step 2: Extract frames
            logger.info(f"Step 2: Extracting {self.num_frames} frames...")
            frames = self.frame_extractor.extract_frames(
                video_path,
                num_frames=self.num_frames,
                method='keyframe'
            )
            
            # Save frames for reference
            if video_name:
                self.frame_extractor.save_frames(frames, video_name)
            
            # Step 3: Perform visual analysis
            logger.info("Step 3: Analyzing frames with CNN...")
            product_results = self.product_classifier.identify_product_from_frames(frames)
            
            # Step 4: Extract text and keywords
            logger.info("Step 4: Extracting text and keywords...")
            keyword_results = self.keyword_extractor.extract_keywords_from_frames(frames)
            
            # Step 5: Identify brand and model from text
            logger.info("Step 5: Identifying brand and model...")
            all_text = ' '.join(keyword_results['high_confidence_text'])
            brand_info = self.brand_identifier.identify_brand_and_model(all_text)
            
            # Also identify from video title and description
            if video_info.get('title'):
                title_brand = self.brand_identifier.identify_brand_and_model(
                    video_info['title']
                )
                if title_brand['brand'] and not brand_info['brand']:
                    brand_info = title_brand
            
            # Step 6: Analyze content
            logger.info("Step 6: Analyzing video content...")
            content_analysis = self.content_analyzer.analyze_content(
                keyword_results['high_confidence_text']
            )
            
            # Step 7: Combine all results
            logger.info("Step 7: Generating final report...")
            final_results = self._combine_results(
                video_info=video_info,
                product_results=product_results,
                keyword_results=keyword_results,
                brand_info=brand_info,
                content_analysis=content_analysis
            )
            
            logger.info("Analysis complete!")
            return final_results
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            return {'error': str(e)}
    
    def analyze_multiple_videos(self, urls: List[str]) -> List[Dict]:
        """
        Analyze multiple YouTube videos
        
        Args:
            urls (list): List of YouTube video URLs
            
        Returns:
            list: List of analysis results for each video
        """
        results = []
        
        for i, url in enumerate(urls):
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing video {i+1}/{len(urls)}")
            logger.info(f"{'='*60}\n")
            
            video_name = f"video_{i+1}"
            result = self.analyze_video_url(url, video_name)
            result['video_index'] = i + 1
            result['url'] = url
            results.append(result)
        
        return results
    
    def _combine_results(self, video_info, product_results, keyword_results, 
                        brand_info, content_analysis) -> Dict:
        """Combine all analysis results into a comprehensive report"""
        
        # Determine the main product
        product_name = None
        product_category = None
        
        # Priority: Brand info from text > Visual classification
        if brand_info.get('full_name'):
            product_name = brand_info['full_name']
        elif product_results.get('most_common_product'):
            product_name = product_results['most_common_product']
        
        if product_results.get('most_common_category'):
            product_category = product_results['most_common_category']
        
        # Extract top keywords
        top_keywords = list(keyword_results['keyword_counts'].keys())[:10]
        
        # Generate summary
        summary = self._generate_summary(
            video_info=video_info,
            product_name=product_name,
            product_category=product_category,
            content_type=content_analysis['content_type'],
            keywords=top_keywords
        )
        
        # Compile final report
        report = {
            'video_title': video_info.get('title', 'Unknown'),
            'video_description': video_info.get('description', '')[:200],
            'duration': video_info.get('duration', 0),
            
            'product_identification': {
                'product_name': product_name,
                'brand': brand_info.get('brand'),
                'model': brand_info.get('model'),
                'category': product_category,
                'confidence': product_results.get('confidence', 0.0)
            },
            
            'keywords': {
                'top_keywords': top_keywords,
                'brands_mentioned': [k['keyword'] for k in keyword_results['categorized_keywords'].get('brands', [])],
                'product_types': [k['keyword'] for k in keyword_results['categorized_keywords'].get('product_types', [])],
                'specifications': [k['keyword'] for k in keyword_results['categorized_keywords'].get('specifications', [])][:10],
            },
            
            'content_analysis': {
                'content_type': content_analysis['content_type'],
                'main_topics': content_analysis['main_topics'],
                'sentiment': content_analysis['sentiment'],
                'description': content_analysis['description']
            },
            
            'summary': summary,
            
            'detailed_results': {
                'visual_classification': product_results,
                'text_extraction': keyword_results,
                'brand_identification': brand_info
            }
        }
        
        return report
    
    def _generate_summary(self, video_info, product_name, product_category, 
                         content_type, keywords) -> str:
        """Generate a human-readable summary of the analysis"""
        
        parts = []
        
        # Video title
        if video_info.get('title'):
            parts.append(f"Video: '{video_info['title']}'")
        
        # Product identification
        if product_name and product_category:
            parts.append(
                f"This is a {content_type} video about the {product_name}, "
                f"which is categorized as a {product_category}."
            )
        elif product_name:
            parts.append(
                f"This is a {content_type} video about the {product_name}."
            )
        elif product_category:
            parts.append(
                f"This is a {content_type} video about a {product_category} product."
            )
        else:
            parts.append(f"This is a {content_type} video about a tech product.")
        
        # Keywords
        if keywords:
            keyword_str = ', '.join(keywords[:5])
            parts.append(f"Key topics include: {keyword_str}.")
        
        return ' '.join(parts)
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save analysis results to a JSON file
        
        Args:
            results (dict): Analysis results
            output_file (str): Path to output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_summary(self, results: Dict):
        """Print a formatted summary of the analysis"""
        
        print("\n" + "="*70)
        print("VIDEO ANALYSIS SUMMARY")
        print("="*70 + "\n")
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Video info
        print(f"📹 Video Title: {results.get('video_title', 'N/A')}")
        print(f"⏱️  Duration: {results.get('duration', 0)} seconds\n")
        
        # Product identification
        prod_id = results.get('product_identification', {})
        print("🏷️  PRODUCT IDENTIFICATION:")
        print(f"   Product: {prod_id.get('product_name', 'Not identified')}")
        print(f"   Brand: {prod_id.get('brand', 'Not identified')}")
        print(f"   Model: {prod_id.get('model', 'Not identified')}")
        print(f"   Category: {prod_id.get('category', 'Not identified')}")
        print(f"   Confidence: {prod_id.get('confidence', 0):.1%}\n")
        
        # Keywords
        kw = results.get('keywords', {})
        print("🔑 TOP KEYWORDS:")
        for keyword in kw.get('top_keywords', [])[:10]:
            print(f"   • {keyword}")
        print()
        
        # Brands and specs
        if kw.get('brands_mentioned'):
            print(f"🏢 Brands Mentioned: {', '.join(kw['brands_mentioned'][:5])}")
        if kw.get('specifications'):
            print(f"⚙️  Specifications: {', '.join(kw['specifications'][:5])}")
        print()
        
        # Content analysis
        content = results.get('content_analysis', {})
        print("📊 CONTENT ANALYSIS:")
        print(f"   Type: {content.get('content_type', 'N/A')}")
        print(f"   Topics: {', '.join(content.get('main_topics', []))}")
        print(f"   Sentiment: {content.get('sentiment', 'N/A')}\n")
        
        # Summary
        print("📝 SUMMARY:")
        print(f"   {results.get('summary', 'No summary available')}\n")
        
        print("="*70 + "\n")


def main():
    """Main function to run the video analyzer"""
    
    print("\n" + "="*70)
    print("YOUTUBE VIDEO ANALYZER - AI-POWERED PRODUCT IDENTIFICATION")
    print("="*70 + "\n")
    
    # Initialize analyzer
    device = 'cuda' if __name__ == '__main__' else 'cpu'
    analyzer = VideoAnalyzer(device=device, num_frames=12)
    
    # Example URLs
    video_urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM"
    ]
    
    print(f"📺 Analyzing {len(video_urls)} YouTube videos...\n")
    
    # Analyze videos
    results = analyzer.analyze_multiple_videos(video_urls)
    
    # Print summaries
    for i, result in enumerate(results):
        analyzer.print_summary(result)
        
        # Save individual results
        output_file = f"results_video_{i+1}.json"
        analyzer.save_results(result, output_file)
    
    # Save combined results
    analyzer.save_results(
        {'videos': results, 'total_analyzed': len(results)},
        'all_results.json'
    )
    
    print("✅ Analysis complete! Results saved to JSON files.")


if __name__ == "__main__":
    main()
