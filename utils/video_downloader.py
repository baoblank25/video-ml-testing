"""
YouTube Video Downloader Utility
Downloads YouTube shorts and videos for analysis
"""

import os
import yt_dlp
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDownloader:
    """Download YouTube videos for processing"""
    
    def __init__(self, output_dir='data'):
        """
        Initialize the video downloader
        
        Args:
            output_dir (str): Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_video(self, url, filename=None):
        """
        Download a YouTube video
        
        Args:
            url (str): YouTube video URL
            filename (str, optional): Custom filename for the video
            
        Returns:
            str: Path to the downloaded video
        """
        try:
            if filename is None:
                output_template = str(self.output_dir / '%(id)s.%(ext)s')
            else:
                output_template = str(self.output_dir / f'{filename}.%(ext)s')
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
            }
            
            logger.info(f"Downloading video from: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get('id', 'video')
                ext = info.get('ext', 'mp4')
                
                if filename is None:
                    video_path = self.output_dir / f'{video_id}.{ext}'
                else:
                    video_path = self.output_dir / f'{filename}.{ext}'
                
                logger.info(f"Video downloaded successfully: {video_path}")
                return str(video_path)
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def download_multiple_videos(self, urls):
        """
        Download multiple YouTube videos
        
        Args:
            urls (list): List of YouTube video URLs
            
        Returns:
            list: List of paths to downloaded videos
        """
        video_paths = []
        
        for i, url in enumerate(urls):
            try:
                filename = f'video_{i+1}'
                path = self.download_video(url, filename)
                video_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to download video {i+1}: {str(e)}")
                
        return video_paths
    
    def get_video_info(self, url):
        """
        Extract video metadata without downloading
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Video metadata
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                metadata = {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'uploader': info.get('uploader', ''),
                    'tags': info.get('tags', []),
                    'categories': info.get('categories', []),
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting video info: {str(e)}")
            return {}


if __name__ == "__main__":
    # Example usage
    downloader = VideoDownloader()
    
    # Example URLs (replace with actual URLs)
    test_urls = [
        "https://www.youtube.com/shorts/MzIen6fSQwA",
        "https://www.youtube.com/shorts/9tMTeEMrpOM"
    ]
    
    for url in test_urls:
        info = downloader.get_video_info(url)
        print(f"Video: {info.get('title', 'Unknown')}")
        print(f"Description: {info.get('description', 'No description')[:100]}...")
