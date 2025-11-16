# 🎯 HOW TO RUN YOUR AI VIDEO ANALYZER

## ⚡ QUICK START - Choose Your Version

### 🚀 Option 1: ENHANCED VERSION (Recommended!)

**Features:**
- ✅ Ensemble CNN with 3 models (Like Google Lens!)
- ✅ Google Search Integration
- ✅ Color & Tone Analysis
- ✅ Event Predictions  
- ✅ Comprehensive Summaries
- ✅ Beautiful console output

**Run Command:**
```powershell
.\.venv\Scripts\python.exe RUN_ENHANCED.py
```

**OR:**
```powershell
.\.venv\Scripts\python.exe enhanced_analyzer.py
```

---

### 📦 Option 2: ORIGINAL VERSION (Faster)

**Features:**
- ✅ Single CNN Model (ResNet50)
- ✅ Product Identification
- ✅ Keyword Extraction
- ✅ Basic Analysis

**Run Command:**
```powershell
.\.venv\Scripts\python.exe analyze_videos.py
```

---

## 📊 WHAT YOU'LL GET

### Enhanced Version Output:
```
🤖 PRODUCT IDENTIFICATION (Ensemble CNN)
   Product:    Dell XPS 15 Laptop
   Brand:      Dell
   Confidence: 89.5%

🔍 GOOGLE SEARCH VERIFICATION
   Status: ✓ Verified
   
🎨 COLOR & TONE ANALYSIS
   Overall Tone:    Neutral/Professional
   Dominant Colors: White, Gray, Blue
   Mood:           professional, clean, minimal

🎉 SUITABLE EVENTS
   • Business Conference
   • Tech Expo
   • Professional Trade Show

📖 COMPREHENSIVE SUMMARY
   [Full narrative description...]
```

### Original Version Output:
```
🏷️ PRODUCT IDENTIFICATION:
   Product: laptop computer
   Category: Laptops
   Confidence: 75.0%

🔑 TOP KEYWORDS:
   • dell, xps, laptop, 15

📊 CONTENT ANALYSIS:
   Type: review
   Sentiment: positive
```

---

## 🎛️ CUSTOMIZATION

### Change Number of Frames
```python
# Edit enhanced_analyzer.py or analyze_videos.py
analyzer = EnhancedVideoAnalyzer(device='cpu', num_frames=15)  # More frames = more accurate
```

### Use Different Videos
```python
# Edit the video_urls list at the bottom of the file
video_urls = [
    "https://www.youtube.com/shorts/YOUR_VIDEO_1",
    "https://www.youtube.com/shorts/YOUR_VIDEO_2"
]
```

### Enable Google Search (Optional)
```python
# Edit utils/google_search.py
api_key = "YOUR_GOOGLE_API_KEY"
search_engine_id = "YOUR_SEARCH_ENGINE_ID"
```
See `ENHANCED_FEATURES.md` for instructions.

---

## ⚠️ REQUIREMENTS

All packages should already be installed. If you get errors:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🎉 READY TO GO!

**Recommended command to start:**
```powershell
.\.venv\Scripts\python.exe RUN_ENHANCED.py
```

This will:
1. ✅ Analyze both YouTube shorts
2. ✅ Print beautiful results to console  
3. ✅ Save detailed JSON files
4. ✅ Give you everything you asked for!

**Expected time:** 2-5 minutes per video

---

## 📁 OUTPUT FILES

After running, check these files:
- `enhanced_results_video_1.json` - Full analysis of video 1
- `enhanced_results_video_2.json` - Full analysis of video 2  
- Console output - Beautiful formatted results

---

## 💡 TIPS

- **Use GPU** if available (10x faster)
- **More frames** = better accuracy (but slower)
- **Enhanced version** gives MUCH more information
- **Original version** is faster if you're in a hurry

---

**You're all set! Just run the command above! 🚀**
