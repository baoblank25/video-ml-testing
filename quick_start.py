"""
QUICK START GUIDE
=================

Run this script to quickly analyze your YouTube shorts!
"""

print("""
╔══════════════════════════════════════════════════════════════════╗
║         YouTube Video Analyzer - Quick Start Guide              ║
║              AI-Powered Product Identification                   ║
╚══════════════════════════════════════════════════════════════════╝

This AI system will analyze your YouTube shorts and provide:

✅ Product identification (e.g., "Dell XPS 15 Laptop")
✅ Brand and model recognition
✅ Category classification (Laptops, Phones, etc.)
✅ Keywords extraction
✅ Technical specifications
✅ Content analysis (review, unboxing, etc.)

Your YouTube shorts to analyze:
1. https://www.youtube.com/shorts/MzIen6fSQwA
2. https://www.youtube.com/shorts/9tMTeEMrpOM

═══════════════════════════════════════════════════════════════════

STEP 1: Choose your device
---------------------------
""")

import torch

if torch.cuda.is_available():
    print("✓ CUDA GPU detected! Using GPU for faster processing.")
    device = 'cuda'
else:
    print("✓ Using CPU (slower but works fine)")
    device = 'cpu'

print(f"\nDevice selected: {device}")

print("""
STEP 2: Initialize the AI system
----------------------------------
Loading models (this may take a minute on first run)...
""")

from analyze_videos import VideoAnalyzer

try:
    # Initialize analyzer
    analyzer = VideoAnalyzer(device=device, num_frames=10)
    print("✓ AI system initialized successfully!\n")
    
    print("""
STEP 3: Analyzing your videos
-------------------------------
Processing your YouTube shorts...
This will take a few minutes depending on video length and your hardware.
""")
    
    # Your YouTube shorts
    urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM"
    ]
    
    # Analyze all videos
    results = analyzer.analyze_multiple_videos(urls)
    
    print("""
STEP 4: Results
----------------
""")
    
    # Print results for each video
    for i, result in enumerate(results):
        analyzer.print_summary(result)
        
        # Save individual results
        output_file = f"video_{i+1}_results.json"
        analyzer.save_results(result, output_file)
        print(f"💾 Results saved to: {output_file}\n")
    
    # Save combined results
    analyzer.save_results(
        {'videos': results, 'total_analyzed': len(results)},
        'all_results.json'
    )
    
    print("""
═══════════════════════════════════════════════════════════════════
✅ ANALYSIS COMPLETE!
═══════════════════════════════════════════════════════════════════

Results have been saved to JSON files:
• video_1_results.json
• video_2_results.json
• all_results.json

You can now:
1. Open the JSON files to see detailed results
2. Run this script again with different videos
3. Check the README.md for advanced usage
4. Customize the code for your specific needs

═══════════════════════════════════════════════════════════════════

💡 Tips:
• Use GPU (CUDA) for 10x faster processing
• Increase num_frames for more accurate results
• Check 'example_usage.py' for more examples
• See 'README.md' for full documentation

═══════════════════════════════════════════════════════════════════
""")

except Exception as e:
    print(f"""
❌ ERROR: {str(e)}

Troubleshooting:
1. Make sure all packages are installed:
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

2. Check your internet connection (for downloading videos)

3. Try running with CPU if GPU fails:
   Edit this file and set device = 'cpu'

4. Check the README.md for more help

═══════════════════════════════════════════════════════════════════
""")
