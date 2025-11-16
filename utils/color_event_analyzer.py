"""
Color Tone Analysis and Event Prediction
Analyzes color palettes and predicts suitable events
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import Counter
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorAnalyzer:
    """
    Analyze color tones and palettes in video frames
    """
    
    def __init__(self):
        """Initialize color analyzer"""
        # Define color ranges in HSV
        self.color_ranges = {
            'Red': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'Red2': {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            'Orange': {'lower': np.array([10, 50, 50]), 'upper': np.array([25, 255, 255])},
            'Yellow': {'lower': np.array([25, 50, 50]), 'upper': np.array([35, 255, 255])},
            'Green': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
            'Cyan': {'lower': np.array([85, 50, 50]), 'upper': np.array([95, 255, 255])},
            'Blue': {'lower': np.array([95, 50, 50]), 'upper': np.array([130, 255, 255])},
            'Purple': {'lower': np.array([130, 50, 50]), 'upper': np.array([160, 255, 255])},
            'Pink': {'lower': np.array([160, 50, 50]), 'upper': np.array([170, 255, 255])},
            'White': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'Black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])},
            'Gray': {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 30, 200])}
        }
        
        # Tone associations
        self.tone_moods = {
            'Red': ['energetic', 'passionate', 'bold', 'powerful', 'exciting'],
            'Orange': ['vibrant', 'creative', 'warm', 'friendly', 'enthusiastic'],
            'Yellow': ['cheerful', 'optimistic', 'bright', 'sunny', 'happy'],
            'Green': ['natural', 'calm', 'fresh', 'growth', 'balanced'],
            'Blue': ['professional', 'trustworthy', 'calm', 'corporate', 'stable'],
            'Purple': ['luxury', 'elegant', 'sophisticated', 'royal', 'creative'],
            'Pink': ['feminine', 'playful', 'romantic', 'soft', 'gentle'],
            'White': ['clean', 'minimal', 'pure', 'modern', 'simple'],
            'Black': ['elegant', 'sophisticated', 'premium', 'professional', 'bold'],
            'Gray': ['neutral', 'professional', 'balanced', 'modern', 'subtle']
        }
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze color distribution in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Color analysis results
        """
        # Convert to RGB for display purposes
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect colors
        color_percentages = {}
        total_pixels = frame.shape[0] * frame.shape[1]
        
        for color_name, ranges in self.color_ranges.items():
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            percentage = (np.count_nonzero(mask) / total_pixels) * 100
            
            # Combine Red and Red2
            if color_name == 'Red2':
                if 'Red' in color_percentages:
                    color_percentages['Red'] += percentage
                continue
            
            color_percentages[color_name] = percentage
        
        # Get dominant colors using K-means
        dominant_colors = self._extract_dominant_colors(frame_rgb, n_colors=5)
        
        return {
            'color_percentages': color_percentages,
            'dominant_colors': dominant_colors,
            'primary_color': max(color_percentages.items(), key=lambda x: x[1])[0],
            'primary_percentage': max(color_percentages.values())
        }
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[Tuple]:
        """
        Extract dominant colors using K-means clustering
        
        Args:
            image: Input image (RGB)
            n_colors: Number of dominant colors to extract
            
        Returns:
            list: List of (color_rgb, percentage) tuples
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Sample pixels for faster processing
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Calculate percentages
        total = len(labels)
        dominant = []
        for i in range(n_colors):
            percentage = (label_counts[i] / total) * 100
            dominant.append({
                'rgb': tuple(colors[i]),
                'hex': '#{:02x}{:02x}{:02x}'.format(*colors[i]),
                'percentage': percentage
            })
        
        # Sort by percentage
        dominant.sort(key=lambda x: x['percentage'], reverse=True)
        
        return dominant
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze color tones across multiple video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            dict: Aggregated color analysis
        """
        logger.info(f"Analyzing color tones across {len(frames)} frames...")
        
        all_analyses = []
        color_accumulator = {color: [] for color in self.color_ranges.keys() if color != 'Red2'}
        
        for frame in frames:
            analysis = self.analyze_frame(frame)
            all_analyses.append(analysis)
            
            # Accumulate color percentages
            for color, percentage in analysis['color_percentages'].items():
                if color in color_accumulator:
                    color_accumulator[color].append(percentage)
        
        # Calculate average color distribution
        avg_colors = {}
        for color, percentages in color_accumulator.items():
            if percentages:
                avg_colors[color] = np.mean(percentages)
        
        # Sort by percentage
        sorted_colors = sorted(avg_colors.items(), key=lambda x: x[1], reverse=True)
        
        # Get top colors
        top_colors = [color for color, pct in sorted_colors[:3] if pct > 5]
        
        # Determine overall tone
        overall_tone = self._determine_overall_tone(top_colors, avg_colors)
        
        # Get mood descriptors
        moods = self._get_mood_descriptors(top_colors)
        
        # Aggregate dominant colors across frames
        all_dominant = []
        for analysis in all_analyses:
            all_dominant.extend(analysis['dominant_colors'])
        
        # Get most common dominant colors
        color_hex_counts = Counter([c['hex'] for c in all_dominant])
        top_dominant = []
        for hex_color, count in color_hex_counts.most_common(5):
            # Find RGB for this hex
            for c in all_dominant:
                if c['hex'] == hex_color:
                    top_dominant.append({
                        'hex': hex_color,
                        'rgb': c['rgb'],
                        'frequency': count / len(frames)
                    })
                    break
        
        result = {
            'average_color_distribution': dict(sorted_colors),
            'top_colors': top_colors,
            'overall_tone': overall_tone,
            'mood_descriptors': moods,
            'dominant_palette': top_dominant[:5],
            'color_scheme': self._identify_color_scheme(top_colors)
        }
        
        return result
    
    def _determine_overall_tone(self, top_colors: List[str], color_dist: Dict) -> str:
        """Determine the overall color tone of the video"""
        if not top_colors:
            return 'Neutral'
        
        # Check for specific tone combinations
        if 'White' in top_colors or 'Gray' in top_colors:
            if color_dist.get('White', 0) > 30:
                return 'Bright/Minimal'
            elif color_dist.get('Black', 0) > 20:
                return 'High Contrast'
            else:
                return 'Neutral/Professional'
        
        if 'Black' in top_colors:
            return 'Dark/Elegant'
        
        # Warm vs cool
        warm_colors = {'Red', 'Orange', 'Yellow', 'Pink'}
        cool_colors = {'Blue', 'Cyan', 'Green', 'Purple'}
        
        warm_count = sum(1 for c in top_colors if c in warm_colors)
        cool_count = sum(1 for c in top_colors if c in cool_colors)
        
        if warm_count > cool_count:
            return 'Warm'
        elif cool_count > warm_count:
            return 'Cool'
        else:
            return 'Balanced'
    
    def _get_mood_descriptors(self, colors: List[str]) -> List[str]:
        """Get mood descriptors based on colors"""
        moods = []
        for color in colors[:3]:  # Top 3 colors
            if color in self.tone_moods:
                moods.extend(self.tone_moods[color][:2])
        
        # Remove duplicates and return top 5
        return list(dict.fromkeys(moods))[:5]
    
    def _identify_color_scheme(self, colors: List[str]) -> str:
        """Identify the color scheme type"""
        if len(colors) < 2:
            return 'Monochromatic'
        
        if 'White' in colors and 'Black' in colors:
            return 'Monochrome (Black & White)'
        
        if all(c in ['White', 'Gray', 'Black'] for c in colors):
            return 'Grayscale/Neutral'
        
        warm = {'Red', 'Orange', 'Yellow', 'Pink'}
        cool = {'Blue', 'Cyan', 'Green', 'Purple'}
        
        warm_count = sum(1 for c in colors if c in warm)
        cool_count = sum(1 for c in colors if c in cool)
        
        if warm_count > 0 and cool_count > 0:
            return 'Complementary/Mixed'
        elif warm_count > 0:
            return 'Warm Palette'
        elif cool_count > 0:
            return 'Cool Palette'
        
        return 'Varied'


class EventPredictor:
    """
    Predict suitable events based on product and color analysis
    """
    
    def __init__(self):
        """Initialize event predictor"""
        # Event associations
        self.product_events = {
            'Laptops': ['Business Conference', 'Tech Expo', 'Product Launch', 'Workspace Setup', 'Home Office'],
            'Smartphones': ['Mobile Tech Show', 'Product Unveiling', 'Consumer Electronics Show', 'Lifestyle Event'],
            'Tablets': ['Education Fair', 'Digital Art Exhibition', 'Business Meeting', 'E-Learning Conference'],
            'Headphones': ['Music Festival', 'Audio Tech Show', 'Gaming Event', 'Podcast Convention'],
            'Smartwatch': ['Fitness Expo', 'Wearable Tech Conference', 'Health & Wellness Fair', 'Sports Event'],
            'Gaming': ['Gaming Convention', 'E-Sports Tournament', 'Tech Gaming Show', 'LAN Party'],
            'Audio Devices': ['Music Event', 'Home Theater Exhibition', 'Audio Tech Fair'],
            'Beverages': ['Food & Beverage Expo', 'Wellness Fair', 'Culinary Event', 'Tea Ceremony', 'Coffee Festival'],
            'Drinkware': ['Kitchen & Home Show', 'Lifestyle Fair', 'Gift Show', 'Home Goods Exhibition']
        }
        
        # Color tone events
        self.tone_events = {
            'Bright/Minimal': ['Modern Art Gallery', 'Design Conference', 'Minimalist Exhibition', 'Tech Showcase'],
            'Dark/Elegant': ['Luxury Event', 'Premium Product Launch', 'High-End Fashion Show', 'Executive Gala'],
            'Warm': ['Cozy Home Show', 'Autumn Fair', 'Comfort Product Launch', 'Family Event'],
            'Cool': ['Tech Conference', 'Professional Business Event', 'Corporate Launch', 'Innovation Summit'],
            'High Contrast': ['Bold Product Launch', 'Statement Event', 'Artistic Exhibition', 'Modern Design Show'],
            'Neutral/Professional': ['Business Conference', 'Corporate Event', 'Professional Trade Show', 'B2B Expo']
        }
        
        # Season associations based on colors
        self.seasonal_events = {
            'Red': ['Christmas Event', 'Valentine\'s Day', 'Chinese New Year', 'Holiday Shopping'],
            'Orange': ['Halloween Event', 'Autumn Fair', 'Thanksgiving', 'Fall Festival'],
            'Yellow': ['Summer Festival', 'Spring Event', 'Outdoor Fair', 'Sunshine Festival'],
            'Green': ['St. Patrick\'s Day', 'Eco-Fair', 'Earth Day Event', 'Nature Exhibition'],
            'Blue': ['Winter Event', 'Ocean Festival', 'Summer Beach Event', 'Corporate Event'],
            'Pink': ['Valentine\'s Day', 'Breast Cancer Awareness', 'Spring Fashion', 'Beauty Fair'],
            'Purple': ['Luxury Event', 'Royal Exhibition', 'Creative Arts Fair', 'Innovation Event'],
            'White': ['Winter Wonderland', 'Wedding Expo', 'Clean Beauty Fair', 'Minimalist Show'],
            'Black': ['Black Friday Sale', 'Luxury Event', 'Premium Launch', 'Formal Gala']
        }
    
    def predict_events(self, product_category: str, color_analysis: Dict) -> Dict:
        """
        Predict suitable events based on product and colors
        
        Args:
            product_category: Category of the product
            color_analysis: Color tone analysis results
            
        Returns:
            dict: Event predictions
        """
        logger.info("Predicting suitable events...")
        
        # Get events for product
        product_events = self.product_events.get(product_category, ['General Product Expo', 'Trade Show'])
        
        # Get events for tone
        overall_tone = color_analysis.get('overall_tone', 'Neutral')
        tone_events = self.tone_events.get(overall_tone, ['General Event', 'Product Showcase'])
        
        # Get seasonal events based on top colors
        seasonal = []
        for color in color_analysis.get('top_colors', [])[:2]:
            if color in self.seasonal_events:
                seasonal.extend(self.seasonal_events[color][:2])
        
        # Combine and deduplicate
        all_events = product_events[:3] + tone_events[:2] + seasonal[:2]
        unique_events = list(dict.fromkeys(all_events))[:8]
        
        # Categorize events
        categorized = {
            'primary_events': unique_events[:3],
            'secondary_events': unique_events[3:6],
            'seasonal_events': list(dict.fromkeys(seasonal))[:3],
            'all_suitable_events': unique_events
        }
        
        # Generate event description
        description = self._generate_event_description(
            product_category,
            color_analysis,
            categorized
        )
        
        return {
            **categorized,
            'description': description,
            'best_match': unique_events[0] if unique_events else 'General Event'
        }
    
    def _generate_event_description(self, product: str, colors: Dict, events: Dict) -> str:
        """Generate human-readable event description"""
        tone = colors.get('overall_tone', 'neutral')
        moods = colors.get('mood_descriptors', [])
        best_event = events['primary_events'][0] if events['primary_events'] else 'event'
        
        mood_str = ', '.join(moods[:3]) if moods else 'appealing'
        
        description = (
            f"This {product} with its {tone.lower()} color tone and {mood_str} aesthetic "
            f"would be perfect for events like {best_event}. "
        )
        
        if events['seasonal_events']:
            description += f"It's particularly suitable for {events['seasonal_events'][0]}. "
        
        return description


if __name__ == "__main__":
    print("Color Analyzer and Event Predictor initialized")
    
    analyzer = ColorAnalyzer()
    predictor = EventPredictor()
    
    print("\nColor Tone Moods:")
    for color, moods in list(analyzer.tone_moods.items())[:5]:
        print(f"  {color}: {', '.join(moods)}")
    
    print("\nEvent Categories:")
    for category in list(predictor.product_events.keys())[:5]:
        print(f"  {category}")
