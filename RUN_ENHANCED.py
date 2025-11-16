"""
ENHANCED AI VIDEO ANALYZER

Analyzes YouTube videos with:
- Ensemble CNN (3 models)
- Google Search Integration  
- Color & Tone Analysis
- Event Predictions
- Comprehensive Summaries
"""

print("""
================================================================================
    ENHANCED AI VIDEO ANALYZER - READY TO RUN!
================================================================================

This will analyze your YouTube shorts with:

  * Ensemble CNN (ResNet + EfficientNet + Vision Transformer)
  * Google Search Verification
  * Color & Tone Analysis
  * Event Predictions
  * Comprehensive Video Summary

All results will be printed to console in clean format!

================================================================================
""")

import sys

try:
    from enhanced_analyzer import EnhancedVideoAnalyzer
    import torch
    
    print("\nInitializing Enhanced Analyzer...\n")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("* GPU detected! Using CUDA for faster processing.\n")
    else:
        print("* Using CPU (will take a bit longer, but works great!)\n")
    
    # Initialize
    analyzer = EnhancedVideoAnalyzer(device=device, num_frames=12)
    
    # Your videos
    video_urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM",
        "https://www.youtube.com/shorts/5BjH0DxidFI"
    ]
    
    print(f"Analyzing {len(video_urls)} YouTube videos...\n")
    print("This will take 2-5 minutes per video (worth the wait!)\n")
    print("="*80 + "\n")
    
    # Analyze each video
    for i, url in enumerate(video_urls, 1):
        print(f"\n{'='*80}")
        print(f"VIDEO {i}/{len(video_urls)}")
        print(f"{'='*80}\n")
        
        result = analyzer.analyze_video_url(url, f"video_{i}")
        
        # Print beautiful results to console
        analyzer.print_enhanced_results(result)
        
        # Save to file
        import json
        filename = f"enhanced_results_video_{i}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved detailed results to: {filename}\n")
    
    print("\n" + "="*80)
    print("ALL DONE! Check the console output above for detailed analysis.")
    print("="*80)
    print("\nResults saved to:")
    print("   * enhanced_results_video_1.json")
    print("   * enhanced_results_video_2.json")
    print("   * enhanced_results_video_3.json")
    print("\nSummary of features analyzed:")
    print("   * Product identification with 3-model ensemble")
    print("   * Brand and model detection")
    print("   * Color tone and palette analysis")
    print("   * Mood and aesthetic description")
    print("   * Event predictions (8+ suitable events)")
    print("   * Target audience identification")
    print("   * Google search verification")
    print("   * Comprehensive video summary")
    print("   * Technical specifications")
    print("   * Keywords and sentiment analysis")
    print("\n" + "="*80 + "\n")

except ImportError as e:
    print(f"""
ERROR: Missing dependencies

Please install required packages:

    .venv\\Scripts\\python.exe -m pip install -r requirements.txt

Then run this script again.

Error details: {str(e)}
""")

except Exception as e:
    print(f"""
ERROR: {str(e)}

Troubleshooting:
1. Make sure all packages are installed
2. Check your internet connection (for video downloads)
3. If using GPU, try CPU mode by editing enhanced_analyzer.py
4. Check ENHANCED_FEATURES.md for more help

Full error:
""")
    import traceback
    traceback.print_exc()
