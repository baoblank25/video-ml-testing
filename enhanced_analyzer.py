"""
Enhanced YouTube Video Analyzer with Ensemble Model & Google Search
Complete AI system with multi-model approach, color analysis, and event prediction
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
from utils.google_search import GoogleSearchAPI, ProductEnhancer
from utils.color_event_analyzer import ColorAnalyzer, EventPredictor
from utils.video_summarizer import VideoSummarizer
from utils.smart_product_detector import enhance_product_detection
from models.ensemble_model import EnsembleCNN, EnsembleProductClassifier
from models.product_classifier import ProductBrandIdentifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedVideoAnalyzer:
    """
    Enhanced video analyzer with ensemble CNN, Google Search, and advanced features
    """
    
    def __init__(self, device='cuda', num_frames=15, google_api_key=None, google_engine_id=None):
        """
        Initialize the enhanced video analyzer
        
        Args:
            device (str): Device to use ('cuda' or 'cpu')
            num_frames (int): Number of frames to extract
            google_api_key (str): Google API key (optional)
            google_engine_id (str): Google Custom Search Engine ID (optional)
        """
        print("\n" + "="*80)
        print("INITIALIZING ENHANCED AI VIDEO ANALYZER")
        print("Multi-Model Ensemble • Google Search • Color Analysis • Event Prediction")
        print("="*80 + "\n")
        
        self.device = device
        self.num_frames = num_frames
        
        # Initialize components
        logger.info("🔧 Loading video processing utilities...")
        self.downloader = VideoDownloader(output_dir='data')
        self.frame_extractor = FrameExtractor(output_dir='preprocessed')
        
        logger.info("🤖 Loading Ensemble CNN (Google Lens approach)...")
        print("   Building ensemble with ResNet50, EfficientNet-B3, Vision Transformer...")
        self.ensemble_cnn = EnsembleCNN(device=device)
        
        logger.info("🏷️  Loading product classifier...")
        self.product_classifier = EnsembleProductClassifier(
            ensemble_model=self.ensemble_cnn,
            device=device
        )
        
        logger.info("🔍 Setting up Google Search integration...")
        self.google_api = GoogleSearchAPI(
            api_key=google_api_key,
            search_engine_id=google_engine_id
        )
        self.product_enhancer = ProductEnhancer(self.google_api)
        
        logger.info("📝 Loading text analysis components...")
        self.keyword_extractor = KeywordExtractor(languages=['en'])
        self.content_analyzer = ContentAnalyzer()
        self.brand_identifier = ProductBrandIdentifier()
        
        logger.info("🎨 Loading color analysis system...")
        self.color_analyzer = ColorAnalyzer()
        
        logger.info("🎉 Loading event predictor...")
        self.event_predictor = EventPredictor()
        
        logger.info("📖 Loading video summarizer...")
        self.video_summarizer = VideoSummarizer()
        
        print("\n✅ Enhanced Video Analyzer initialized successfully!")
        print("="*80 + "\n")
    
    def analyze_video_url(self, url: str, video_name: str = None) -> Dict:
        """
        Analyze a YouTube video with all enhanced features
        
        Args:
            url (str): YouTube video URL
            video_name (str, optional): Custom name for the video
            
        Returns:
            dict: Complete enhanced analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING VIDEO: {url}")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: Download video
            print("STEP 1: Downloading video...")
            video_path = self.downloader.download_video(url, filename=video_name)
            video_info = self.downloader.get_video_info(url)
            print(f"Downloaded: {video_info.get('title', 'Video')}\n")
            
            # Step 2: Extract frames
            print(f"STEP 2: Extracting {self.num_frames} key frames...")
            frames = self.frame_extractor.extract_frames(
                video_path,
                num_frames=self.num_frames,
                method='keyframe'
            )
            if video_name:
                self.frame_extractor.save_frames(frames, video_name)
            print(f"Extracted {len(frames)} frames\n")
            
            # Step 3: Ensemble CNN Analysis
            print("STEP 3: Analyzing with Ensemble CNN (3 models)...")
            ensemble_results = self.product_classifier.identify_product_from_frames(frames)
            
            if ensemble_results:
                print(f"Ensemble analysis complete (confidence: {ensemble_results.get('confidence', 0):.1%})\n")
            else:
                print("Ensemble analysis returned no results (will use metadata only)\n")
                ensemble_results = None
            
            # Step 4: Color Analysis
            print("STEP 4: Analyzing color tones and palette...")
            color_analysis = self.color_analyzer.analyze_video_frames(frames)
            print(f"Dominant tones: {', '.join(color_analysis.get('top_colors', [])[:3])}\n")
            
            # Step 5: Text Extraction
            print("STEP 5: Extracting text and keywords (OCR)...")
            keyword_results = self.keyword_extractor.extract_keywords_from_frames(frames)
            print(f"Found {len(keyword_results.get('keyword_counts', {}))} unique keywords\n")
            
            # Step 6: Smart Product Detection (OCR + Metadata + CNN)
            print("STEP 6: Smart product detection (OCR + Metadata + CNN)...")
            smart_detection = enhance_product_detection(video_info, keyword_results, ensemble_results)
            
            if smart_detection.get('brand') and smart_detection.get('model'):
                print(f"Identified: {smart_detection['brand']} {smart_detection['model']} ({smart_detection['confidence']*100:.0f}% confidence)\n")
            elif smart_detection.get('brand'):
                print(f"Brand identified: {smart_detection['brand']}\n")
            else:
                print("Smart detection complete\n")
            
            # Step 7: Google Search Enhancement
            print("STEP 7: Verifying with Google Search...")
            
            # Safe extraction of product info - handle None ensemble_results
            product_name = smart_detection.get('product_name', 'product')
            product_category = smart_detection.get('category', 'General Product')
            
            # Use CNN results if available and smart detection didn't find anything
            if not product_name and ensemble_results and ensemble_results.get('primary_prediction'):
                product_name = ensemble_results['primary_prediction'].get('label', 'product')
            if not product_category and ensemble_results and ensemble_results.get('primary_prediction'):
                product_category = ensemble_results['primary_prediction'].get('category', 'General Product')
            
            detected_product = {
                'product_name': product_name,
                'brand': smart_detection.get('brand'),
                'confidence': smart_detection.get('confidence', 0.5)
            }
            
            enhanced_product = self.product_enhancer.enhance_detection(
                detected_product,
                list(keyword_results.get('keyword_counts', {}).keys())[:5]
            )
            print(f"Google verification: {'Confirmed' if enhanced_product.get('google_verified') else 'Completed'}\n")
            
            # Step 8: Event Prediction
            print("STEP 8: Predicting suitable events...")
            event_predictions = self.event_predictor.predict_events(product_category, color_analysis)
            print(f"Best match: {event_predictions.get('best_match', 'N/A')}\n")
            
            # Step 9: Content Analysis
            print("STEP 9: Analyzing content type and sentiment...")
            content_analysis = self.content_analyzer.analyze_content(
                keyword_results['high_confidence_text']
            )
            print(f"Content type: {content_analysis.get('content_type', 'N/A')} ({content_analysis.get('sentiment', 'neutral')} tone)\n")
            
            # Step 10: Generate Summary
            print("STEP 10: Generating comprehensive summary...")
            
            # Compile all results
            full_analysis = {
                'video_info': video_info,
                'product_identification': {
                    'product_name': smart_detection.get('product_name'),
                    'brand': smart_detection.get('brand'),
                    'model': smart_detection.get('model'),
                    'category': smart_detection.get('category', product_category),
                    'confidence': enhanced_product.get('confidence_adjusted', 0.0),
                    'ensemble_predictions': ensemble_results.get('ensemble_predictions', []) if ensemble_results else []
                },
                'keywords': keyword_results,
                'content_analysis': content_analysis,
                'color_analysis': color_analysis,
                'event_predictions': event_predictions,
                'google_enhanced': enhanced_product
            }
            
            summary = self.video_summarizer.generate_summary(full_analysis)
            print("Summary generated\n")
            
            # Combine everything
            final_results = {
                **full_analysis,
                'summary': summary
            }
            
            print(f"{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}\n")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def print_enhanced_results(self, results: Dict):
        """
        Print enhanced results to console with all new features
        """
        print("\n" + "="*80)
        print("VIDEO ANALYSIS RESULTS - ENHANCED REPORT")
        print("="*80 + "\n")
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return
        
        # Video Information
        video_info = results.get('video_info', {})
        print("VIDEO INFORMATION")
        print("-" * 80)
        print(f"Title: {video_info.get('title', 'N/A')}")
        print(f"Duration: {video_info.get('duration', 0)} seconds")
        if video_info.get('view_count'):
            print(f"Views: {video_info['view_count']:,}")
        print()
        
        # Product Identification (Ensemble)
        product = results.get('product_identification', {})
        print("PRODUCT IDENTIFICATION (Ensemble CNN)")
        print("-" * 80)
        print(f"Product:    {product.get('product_name', 'Not identified')}")
        print(f"Brand:      {product.get('brand', 'Not identified')}")
        print(f"Model:      {product.get('model', 'Not identified')}")
        print(f"Category:   {product.get('category', 'Not identified')}")
        print(f"Confidence: {product.get('confidence', 0):.1%}")
        
        # Show ensemble predictions
        if product.get('ensemble_predictions'):
            print("\nTop Predictions from Ensemble:")
            for i, pred in enumerate(product['ensemble_predictions'][:3], 1):
                print(f"  {i}. {pred.get('label', 'N/A')} ({pred.get('confidence', 0):.1%})")
        print()
        
        # Google Search Enhancement
        google_enhanced = results.get('google_enhanced', {})
        if google_enhanced.get('google_verified'):
            print("GOOGLE SEARCH VERIFICATION")
            print("-" * 80)
            print(f"Status: Verified")
            if google_enhanced.get('enhanced_description'):
                print(f"Description: {google_enhanced['enhanced_description'][:150]}...")
            if google_enhanced.get('specifications_found'):
                print(f"Specifications: {', '.join(google_enhanced['specifications_found'][:5])}")
            print()
        
        # Color & Tone Analysis
        color_analysis = results.get('color_analysis', {})
        if color_analysis:
            print("COLOR & TONE ANALYSIS")
            print("-" * 80)
            print(f"Overall Tone:    {color_analysis.get('overall_tone', 'N/A')}")
            print(f"Dominant Colors: {', '.join(color_analysis.get('top_colors', [])[:3])}")
            print(f"Color Scheme:    {color_analysis.get('color_scheme', 'N/A')}")
            print(f"Mood:            {', '.join(color_analysis.get('mood_descriptors', [])[:5])}")
            
            # Show color palette
            palette = color_analysis.get('dominant_palette', [])
            if palette:
                print("\nDominant Color Palette:")
                for i, color in enumerate(palette[:5], 1):
                    print(f"  {i}. {color['hex']} (RGB: {color['rgb']}) - {color['frequency']:.1%}")
            print()
        
        # Event Predictions
        events = results.get('event_predictions', {})
        if events:
            print("SUITABLE EVENTS & OCCASIONS")
            print("-" * 80)
            print(f"Best Match: {events.get('best_match', 'N/A')}")
            
            if events.get('primary_events'):
                print("\nPrimary Events:")
                for event in events['primary_events'][:3]:
                    print(f"  • {event}")
            
            if events.get('seasonal_events'):
                print("\nSeasonal Events:")
                for event in events['seasonal_events'][:3]:
                    print(f"  • {event}")
            print()
        
        # Keywords
        keywords = results.get('keywords', {})
        if keywords:
            print("KEYWORDS & SPECIFICATIONS")
            print("-" * 80)
            top_kw = list(keywords.get('keyword_counts', {}).keys())[:10]
            print(f"Top Keywords: {', '.join(top_kw)}")
            
            if keywords.get('categorized_keywords'):
                cat_kw = keywords['categorized_keywords']
                if cat_kw.get('brands'):
                    brands = [k['keyword'] for k in cat_kw['brands'][:3]]
                    print(f"Brands:       {', '.join(brands)}")
                if cat_kw.get('specifications'):
                    specs = [k['keyword'] for k in cat_kw['specifications'][:5]]
                    print(f"Specs:        {', '.join(specs)}")
            print()
        
        # Content Analysis
        content = results.get('content_analysis', {})
        if content:
            print("CONTENT ANALYSIS")
            print("-" * 80)
            print(f"Content Type: {content.get('content_type', 'N/A')}")
            print(f"Main Topics:  {', '.join(content.get('main_topics', []))}")
            print(f"Sentiment:    {content.get('sentiment', 'N/A')}")
            print()
        
        # Summary
        summary = results.get('summary', {})
        if summary:
            print("VIDEO SUMMARY")
            print("-" * 80)
            print(summary.get('executive_summary', 'No summary available'))
            print()
            
            if summary.get('mood_and_tone'):
                print("Mood & Tone:")
                print(summary['mood_and_tone'])
                print()
            
            if summary.get('target_audience'):
                print(f"Target Audience: {summary['target_audience']}")
                print()
        
        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")


def main():
    """Main function to run the enhanced video analyzer"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║           ENHANCED AI VIDEO ANALYZER - PRODUCTION VERSION               ║
║                                                                          ║
║  ✓ Ensemble CNN (ResNet50 + EfficientNet + Vision Transformer)         ║
║  ✓ Google Search Integration (Product Verification)                    ║
║  ✓ Advanced Color & Tone Analysis                                      ║
║  ✓ Event Prediction System                                             ║
║  ✓ Comprehensive Video Summarization                                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Initialize analyzer
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    analyzer = EnhancedVideoAnalyzer(device=device, num_frames=12)
    
    # Video URLs
    video_urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM"
    ]
    
    print(f"\n📺 Analyzing {len(video_urls)} YouTube videos with Enhanced AI...\n")
    
    # Analyze videos
    all_results = []
    for i, url in enumerate(video_urls):
        video_name = f"video_{i+1}"
        result = analyzer.analyze_video_url(url, video_name)
        
        # Print enhanced results
        analyzer.print_enhanced_results(result)
        
        # Save results
        output_file = f"enhanced_results_video_{i+1}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Detailed results saved to: {output_file}\n")
        
        all_results.append(result)
    
    # Save combined results
    with open('enhanced_all_results.json', 'w', encoding='utf-8') as f:
        json.dump({'videos': all_results, 'total_analyzed': len(all_results)}, 
                 f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    for i in range(len(all_results)):
        print(f"  • enhanced_results_video_{i+1}.json")
    print(f"  • enhanced_all_results.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
