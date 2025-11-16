"""
Display individual video analysis results with correct product detection
"""

import json
import re
from pathlib import Path
from utils.smart_product_detector import enhance_product_detection


def print_video_analysis(json_file: str, video_number: int):
    """Print complete analysis for one video"""
    
    # Load results
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Get smart detection
    smart_detection = enhance_product_detection(
        results['video_info'],
        results['keywords'],
        results['product_identification']
    )
    
    print(f"\nVIDEO {video_number} ANALYSIS")
    print()
    
    # Video Info
    print("VIDEO INFORMATION")
    print(f"Title:    {results['video_info']['title']}")
    print(f"Duration: {results['video_info']['duration']} seconds")
    print(f"Views:    {results['video_info']['view_count']:,}")
    print(f"Channel:  {results['video_info'].get('uploader', 'N/A')}")
    print()
    
    # Product Identification
    print("\nPRODUCT IDENTIFICATION (SMART DETECTION)")
    print(f"Product:    {smart_detection['product_name']}")
    print(f"Brand:      {smart_detection['brand']}")
    print(f"Model:      {smart_detection['model']}")
    print(f"Category:   {smart_detection['category']}")
    print(f"Confidence: {smart_detection['confidence']*100:.1f}%")
    print(f"Detection:  {', '.join(smart_detection['detection_sources'])}")
    print()
    
    # Color Analysis
    print("\nCOLOR & TONE ANALYSIS")
    color = results['color_analysis']
    print(f"Overall Tone:    {color['overall_tone']}")
    print(f"Dominant Colors: {', '.join(color['top_colors'])}")
    print(f"Color Scheme:    {color['color_scheme']}")
    print(f"Mood:            {', '.join(color['mood_descriptors'][:3])}")
    print()
    
    print("Dominant Color Palette:")
    for i, palette_color in enumerate(color['dominant_palette'][:5], 1):
        hex_code = palette_color['hex']
        rgb = palette_color['rgb']
        freq = palette_color['frequency']*100
        print(f"  {i}. {hex_code} (RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}) - {freq:.1f}%")
    print()
    
    # Events
    print("\nSUITABLE EVENTS & OCCASIONS")
    events = results['event_predictions']
    print(f"Best Match: {events['best_match']}")
    print()
    print("Primary Events:")
    for event in events['primary_events']:
        print(f"  - {event}")
    if events.get('seasonal_events'):
        print()
        print("Seasonal Events:")
        for event in events['seasonal_events']:
            print(f"  - {event}")
    print()
    
    # Keywords
    print("\nKEYWORDS & SPECIFICATIONS")
    keywords_data = results['keywords']
    
    # Extract keywords from video title and tags (most reliable source)
    video_info = results['video_info']
    title_keywords = []
    if video_info.get('title'):
        title_lower = video_info['title'].lower()
        # Extract meaningful words from title
        meaningful_words = re.findall(r'\b[a-z]{4,}\b', title_lower)
        title_keywords = [w for w in meaningful_words if w not in ['this', 'that', 'with', 'from', 'your', 'shorts', 'dont']][:5]
    
    # Extract from tags (very reliable)
    tag_keywords = []
    if video_info.get('tags'):
        # Get first 5 most relevant tags
        for tag in video_info['tags'][:5]:
            # Clean tag
            tag_clean = tag.lower().replace('#', '').strip()
            if len(tag_clean) > 3:
                tag_keywords.append(tag_clean)
    
    # Combine title and tag keywords (skip noisy OCR)
    all_keywords = title_keywords + tag_keywords
    if all_keywords:
        keyword_str = ', '.join(all_keywords[:10])
        print(f"Keywords: {keyword_str}")
    else:
        print(f"Keywords: None detected")
    
    # Specs
    if keywords_data.get('categorized_keywords', {}).get('specifications'):
        specs = [item['keyword'] if isinstance(item, dict) else item 
                for item in keywords_data['categorized_keywords']['specifications']]
        if specs:
            print(f"Specs:    {', '.join(specs)}")
    
    # Models
    if keywords_data.get('categorized_keywords', {}).get('model_identifiers'):
        models = [item['keyword'] if isinstance(item, dict) else item 
                 for item in keywords_data['categorized_keywords']['model_identifiers']]
        if models:
            print(f"Models:   {', '.join(models)}")
    print()
    
    # Content Analysis
    content = results['content_analysis']
    print("\nCONTENT ANALYSIS")
    print(f"Content Type: {content['content_type']}")
    print(f"Sentiment:    {content['sentiment']}")
    print()
    
    # Summary
    summary = results['summary']
    print("\nVIDEO SUMMARY")
    # Print just the executive summary
    print(summary['executive_summary'])
    print()
    print(f"Target Audience: {summary['target_audience']}")
    print(f"\nCORRECTED PRODUCT: {smart_detection['product_name']}")
    print()


def main():
    """Main function to display all videos"""
    
    print("\nYOUTUBE SHORTS ANALYSIS - SEPARATE RESULTS\n")
    
    # Video 1
    if Path('enhanced_results_video_1.json').exists():
        print_video_analysis('enhanced_results_video_1.json', 1)
    
    # Video 2
    if Path('enhanced_results_video_2.json').exists():
        print_video_analysis('enhanced_results_video_2.json', 2)
    
    # Video 3
    if Path('enhanced_results_video_3.json').exists():
        print_video_analysis('enhanced_results_video_3.json', 3)
    
    print("\nANALYSIS DISPLAY COMPLETE\n")


if __name__ == '__main__':
    main()
