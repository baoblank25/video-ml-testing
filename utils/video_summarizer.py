"""
Video Summarizer
Generates comprehensive summaries of video content
"""

import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSummarizer:
    """
    Generate comprehensive video summaries
    """
    
    def __init__(self):
        """Initialize video summarizer"""
        self.content_templates = {
            'review': 'This video provides a {sentiment} review of the {product}, highlighting {topics}.',
            'unboxing': 'This unboxing video showcases the {product}, revealing {topics}.',
            'tutorial': 'This tutorial demonstrates how to use the {product}, focusing on {topics}.',
            'comparison': 'This comparison video analyzes the {product} against alternatives, discussing {topics}.',
            'demonstration': 'This demonstration shows the {product} in action, featuring {topics}.',
            'showcase': 'This showcase presents the {product}, emphasizing {topics}.'
        }
    
    def generate_summary(self, analysis_results: Dict) -> Dict:
        """
        Generate comprehensive video summary
        
        Args:
            analysis_results: Complete analysis results from all components
            
        Returns:
            dict: Comprehensive summary
        """
        logger.info("Generating comprehensive video summary...")
        
        # Extract key information
        video_info = analysis_results.get('video_info', {})
        product_info = analysis_results.get('product_identification', {})
        keywords = analysis_results.get('keywords', {})
        content_analysis = analysis_results.get('content_analysis', {})
        color_analysis = analysis_results.get('color_analysis', {})
        event_predictions = analysis_results.get('event_predictions', {})
        google_results = analysis_results.get('google_enhanced', {})
        
        # Generate different sections of summary
        summary = {
            'title': self._generate_title(video_info, product_info),
            'executive_summary': self._generate_executive_summary(
                product_info, content_analysis, color_analysis
            ),
            'product_details': self._generate_product_details(product_info, google_results),
            'visual_analysis': self._generate_visual_analysis(color_analysis),
            'content_description': self._generate_content_description(content_analysis, keywords),
            'key_features': self._extract_key_features(keywords, google_results),
            'target_audience': self._determine_target_audience(product_info, color_analysis),
            'suitable_events': event_predictions.get('description', 'Various events'),
            'mood_and_tone': self._describe_mood_and_tone(color_analysis, content_analysis),
            'complete_narrative': ''
        }
        
        # Generate complete narrative
        summary['complete_narrative'] = self._generate_narrative(summary, analysis_results)
        
        return summary
    
    def _generate_title(self, video_info: Dict, product_info: Dict) -> str:
        """Generate summary title"""
        video_title = video_info.get('title', '')
        product_name = product_info.get('product_name', 'Product')
        
        if video_title:
            return f"Analysis: {video_title}"
        else:
            return f"Video Analysis: {product_name}"
    
    def _generate_executive_summary(self, product_info: Dict, content: Dict, colors: Dict) -> str:
        """Generate brief executive summary"""
        product_name = product_info.get('product_name', 'the featured product')
        category = product_info.get('category', 'product')
        content_type = content.get('content_type', 'showcase')
        tone = colors.get('overall_tone', 'balanced')
        confidence = product_info.get('confidence', 0.0)
        
        summary = (
            f"This is a {content_type} video featuring {product_name}, "
            f"categorized as {category}. "
        )
        
        if confidence > 0.7:
            summary += f"The product was identified with high confidence ({confidence:.1%}). "
        
        summary += f"The video has a {tone.lower()} color tone. "
        
        return summary
    
    def _generate_product_details(self, product_info: Dict, google_results: Dict) -> str:
        """Generate detailed product description"""
        product_name = product_info.get('product_name', 'Unknown product')
        brand = product_info.get('brand', 'Unknown brand')
        model = product_info.get('model', '')
        category = product_info.get('category', '')
        
        details = f"**Product:** {product_name}\n"
        
        if brand != 'Unknown brand':
            details += f"**Brand:** {brand}\n"
        
        if model:
            details += f"**Model:** {model}\n"
        
        if category:
            details += f"**Category:** {category}\n"
        
        # Add Google-enhanced description
        if google_results.get('enhanced_description'):
            details += f"\n**Description:** {google_results['enhanced_description'][:200]}"
        
        # Add specifications
        if google_results.get('specifications_found'):
            details += f"\n**Specifications:** {', '.join(google_results['specifications_found'][:3])}"
        
        return details
    
    def _generate_visual_analysis(self, color_analysis: Dict) -> str:
        """Generate visual/color analysis description"""
        if not color_analysis:
            return "Visual analysis not available."
        
        tone = color_analysis.get('overall_tone', 'neutral')
        top_colors = color_analysis.get('top_colors', [])
        moods = color_analysis.get('mood_descriptors', [])
        
        analysis = f"The video features a **{tone}** color palette "
        
        if top_colors:
            colors_str = ', '.join(top_colors[:3])
            analysis += f"dominated by {colors_str} tones. "
        
        if moods:
            moods_str = ', '.join(moods[:3])
            analysis += f"This creates a {moods_str} aesthetic. "
        
        # Add color scheme
        scheme = color_analysis.get('color_scheme', '')
        if scheme:
            analysis += f"The overall color scheme is {scheme.lower()}."
        
        return analysis
    
    def _generate_content_description(self, content: Dict, keywords: Dict) -> str:
        """Generate content description"""
        content_type = content.get('content_type', 'video')
        topics = content.get('main_topics', [])
        sentiment = content.get('sentiment', 'neutral')
        
        description = f"This {content_type} has a {sentiment} tone"
        
        if topics:
            topics_str = ', '.join(topics[:3])
            description += f" and focuses on {topics_str}"
        
        description += ". "
        
        # Add keywords
        top_keywords = keywords.get('top_keywords', [])
        if top_keywords:
            kw_str = ', '.join(top_keywords[:5])
            description += f"Key topics include: {kw_str}."
        
        return description
    
    def _extract_key_features(self, keywords: Dict, google_results: Dict) -> List[str]:
        """Extract key features mentioned"""
        features = []
        
        # From specifications
        specs = keywords.get('specifications', [])
        features.extend([s['keyword'] for s in specs[:5]] if isinstance(specs, list) and specs and isinstance(specs[0], dict) else specs[:5] if specs else [])
        
        # From Google results
        if google_results.get('specifications_found'):
            features.extend(google_results['specifications_found'][:3])
        
        # Deduplicate
        unique_features = list(dict.fromkeys(features))
        
        return unique_features[:8]
    
    def _determine_target_audience(self, product_info: Dict, color_analysis: Dict) -> str:
        """Determine target audience based on product and colors"""
        category = product_info.get('category', '').lower()
        moods = color_analysis.get('mood_descriptors', [])
        
        audiences = []
        
        # Category-based
        if 'laptop' in category or 'computer' in category:
            audiences.extend(['Professionals', 'Students', 'Content Creators'])
        elif 'phone' in category or 'smartphone' in category:
            audiences.extend(['General Consumers', 'Tech Enthusiasts', 'Mobile Users'])
        elif 'gaming' in category:
            audiences.extend(['Gamers', 'E-sports Enthusiasts', 'Streaming Community'])
        elif 'beverage' in category or 'drink' in category:
            audiences.extend(['Health-Conscious Consumers', 'Beverage Enthusiasts', 'Lifestyle Seekers'])
        
        # Mood-based
        if any(mood in ['luxury', 'elegant', 'sophisticated', 'premium'] for mood in moods):
            audiences.append('Premium Buyers')
        if any(mood in ['professional', 'corporate', 'business'] for mood in moods):
            audiences.append('Business Professionals')
        if any(mood in ['creative', 'artistic', 'vibrant'] for mood in moods):
            audiences.append('Creative Professionals')
        
        if not audiences:
            audiences = ['General Consumers', 'Product Enthusiasts']
        
        # Deduplicate
        unique_audiences = list(dict.fromkeys(audiences))[:4]
        
        return ', '.join(unique_audiences)
    
    def _describe_mood_and_tone(self, color_analysis: Dict, content: Dict) -> str:
        """Describe the mood and tone"""
        moods = color_analysis.get('mood_descriptors', [])
        tone = color_analysis.get('overall_tone', 'neutral')
        sentiment = content.get('sentiment', 'neutral')
        
        description = f"The video projects a **{sentiment}** sentiment with a **{tone}** visual tone. "
        
        if moods:
            moods_str = ', '.join(moods[:4])
            description += f"The overall mood is {moods_str}, "
        
        description += "making it visually appealing and engaging to viewers."
        
        return description
    
    def _generate_narrative(self, summary: Dict, full_analysis: Dict) -> str:
        """Generate complete narrative summary"""
        narrative_parts = []
        
        # Opening
        narrative_parts.append(summary['executive_summary'])
        narrative_parts.append("")
        
        # Product details
        narrative_parts.append("**PRODUCT INFORMATION**")
        narrative_parts.append(summary['product_details'])
        narrative_parts.append("")
        
        # Visual analysis
        narrative_parts.append("**VISUAL ANALYSIS**")
        narrative_parts.append(summary['visual_analysis'])
        narrative_parts.append("")
        
        # Content
        narrative_parts.append("**CONTENT ANALYSIS**")
        narrative_parts.append(summary['content_description'])
        narrative_parts.append("")
        
        # Key features
        if summary['key_features']:
            narrative_parts.append("**KEY FEATURES MENTIONED**")
            for feature in summary['key_features']:
                narrative_parts.append(f"  • {feature}")
            narrative_parts.append("")
        
        # Mood and tone
        narrative_parts.append("**MOOD & TONE**")
        narrative_parts.append(summary['mood_and_tone'])
        narrative_parts.append("")
        
        # Target audience
        narrative_parts.append(f"**TARGET AUDIENCE:** {summary['target_audience']}")
        narrative_parts.append("")
        
        # Event predictions
        if full_analysis.get('event_predictions'):
            events = full_analysis['event_predictions']
            narrative_parts.append("**SUITABLE EVENTS**")
            narrative_parts.append(summary['suitable_events'])
            
            if events.get('primary_events'):
                narrative_parts.append(f"Primary events: {', '.join(events['primary_events'])}")
            
            if events.get('seasonal_events'):
                narrative_parts.append(f"Seasonal opportunities: {', '.join(events['seasonal_events'])}")
            
            narrative_parts.append("")
        
        # Conclusion
        product_name = full_analysis.get('product_identification', {}).get('product_name', 'this product')
        narrative_parts.append("**CONCLUSION**")
        narrative_parts.append(
            f"This video effectively showcases {product_name} through engaging visuals "
            f"and {summary['content_description'].split('.')[0].lower()}. "
            f"The {summary['visual_analysis'].split('.')[0].lower()} makes it ideal for "
            f"{summary['target_audience'].split(',')[0].lower()} and similar audiences."
        )
        
        return '\n'.join(narrative_parts)
    
    def print_summary(self, summary: Dict):
        """Print formatted summary to console"""
        print("\n" + "="*80)
        print(summary['title'].center(80))
        print("="*80 + "\n")
        
        print(summary['complete_narrative'])
        
        print("\n" + "="*80)


if __name__ == "__main__":
    print("Video Summarizer initialized")
