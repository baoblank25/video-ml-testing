"""
Re-analyze videos with improved smart product detection
This script loads existing JSON results and enhances them with better product identification
"""

import json
import sys
from pathlib import Path
from utils.smart_product_detector import enhance_product_detection

def reanalyze_video_results(json_file: str):
    """
    Re-analyze a video result JSON file with smart detection
    
    Args:
        json_file: Path to the JSON results file
    """
    # Load existing results
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"RE-ANALYZING: {results['video_info']['title']}")
    print(f"{'='*80}\n")
    
    # Extract data for smart detection
    video_info = results['video_info']
    ocr_keywords = results['keywords']
    cnn_results = results['product_identification']
    
    # Run smart detection
    smart_detection = enhance_product_detection(video_info, ocr_keywords, cnn_results)
    
    # Display results
    print("📹 VIDEO INFORMATION")
    print("─" * 80)
    print(f"Title: {video_info['title']}")
    print(f"Duration: {video_info['duration']} seconds")
    print(f"Views: {video_info['view_count']:,}")
    print()
    
    print("🔍 ORIGINAL CNN DETECTION")
    print("─" * 80)
    print(f"Product:    {cnn_results['product_name']}")
    print(f"Brand:      {cnn_results['brand']}")
    print(f"Model:      {cnn_results['model']}")
    print(f"Category:   {cnn_results['category']}")
    print(f"Confidence: {cnn_results['confidence']*100:.1f}%")
    print()
    
    print("✨ SMART ENHANCED DETECTION")
    print("─" * 80)
    print(f"Product:    {smart_detection['product_name']}")
    print(f"Brand:      {smart_detection['brand']}")
    print(f"Model:      {smart_detection['model']}")
    print(f"Category:   {smart_detection['category']}")
    print(f"Confidence: {smart_detection['confidence']*100:.1f}%")
    print(f"Sources:    {', '.join(smart_detection['detection_sources'])}")
    print()
    
    # Show key evidence
    print("📝 KEY EVIDENCE FROM VIDEO")
    print("─" * 80)
    
    # From description
    desc = video_info.get('description', '')
    if len(desc) > 200:
        print(f"Description: {desc[:200]}...")
    else:
        print(f"Description: {desc}")
    print()
    
    # From OCR
    if ocr_keywords.get('high_confidence_text'):
        print(f"OCR Text (High Confidence): {', '.join(ocr_keywords['high_confidence_text'][:10])}")
    
    if ocr_keywords.get('model_identifiers'):
        models = [item['keyword'] if isinstance(item, dict) else item 
                 for item in ocr_keywords['model_identifiers']]
        print(f"Model Identifiers: {', '.join(models)}")
    print()
    
    # From tags
    if video_info.get('tags'):
        relevant_tags = [tag for tag in video_info['tags'][:5]]
        print(f"Relevant Tags: {', '.join(relevant_tags)}")
    print()
    
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print(f"║  ✅ CORRECTED: {smart_detection['product_name']:<60} ║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    return smart_detection


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SMART PRODUCT RE-ANALYSIS")
    print("Using OCR, Video Metadata, and CNN for Accurate Detection")
    print("="*80)
    
    # Find all result JSON files
    json_files = [
        'enhanced_results_video_1.json',
        'enhanced_results_video_2.json',
        'enhanced_results_video_3.json'
    ]
    
    all_results = []
    
    for json_file in json_files:
        if Path(json_file).exists():
            smart_result = reanalyze_video_results(json_file)
            all_results.append(smart_result)
        else:
            print(f"⚠️  File not found: {json_file}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY OF ALL VIDEOS")
    print("="*80)
    
    for i, result in enumerate(all_results, 1):
        print(f"\nVideo {i}:")
        print(f"  Product: {result['product_name']}")
        print(f"  Brand:   {result['brand']}")
        print(f"  Model:   {result['model']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
    
    print("\n" + "="*80)
    print("✅ RE-ANALYSIS COMPLETE!")
    print("="*80)
    print("\nThe smart detector combines:")
    print("  • Video titles and descriptions")
    print("  • OCR text extraction")
    print("  • Video tags and metadata")
    print("  • CNN predictions (as validation)")
    print("\nThis provides much more accurate product identification!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
