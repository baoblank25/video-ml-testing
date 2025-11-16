"""
Video Frame Extraction Utility
Extracts key frames from videos for analysis
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from videos for analysis"""
    
    def __init__(self, output_dir='preprocessed'):
        """
        Initialize the frame extractor
        
        Args:
            output_dir (str): Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self, video_path, num_frames=10, method='uniform'):
        """
        Extract frames from a video
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract
            method (str): 'uniform' or 'keyframe' extraction method
            
        Returns:
            list: List of extracted frames (numpy arrays)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video has {total_frames} frames at {fps} FPS")
            
            if method == 'uniform':
                frames = self._extract_uniform_frames(cap, total_frames, num_frames)
            elif method == 'keyframe':
                frames = self._extract_keyframes(cap, total_frames, num_frames)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def _extract_uniform_frames(self, cap, total_frames, num_frames):
        """Extract frames uniformly distributed across the video"""
        frames = []
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
            else:
                logger.warning(f"Could not read frame at index {idx}")
        
        return frames
    
    def _extract_keyframes(self, cap, total_frames, num_frames):
        """Extract keyframes based on scene changes"""
        frames = []
        prev_frame = None
        frame_diffs = []
        all_frames = []
        
        # First pass: calculate differences between consecutive frames
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            all_frames.append(frame)
            
            if prev_frame is not None:
                # Calculate difference between frames
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray_prev, gray_curr)
                diff_score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
                frame_diffs.append((i, diff_score))
            
            prev_frame = frame
        
        # Select frames with highest differences (scene changes)
        if frame_diffs:
            frame_diffs.sort(key=lambda x: x[1], reverse=True)
            selected_indices = sorted([idx for idx, _ in frame_diffs[:num_frames]])
            
            for idx in selected_indices:
                if idx < len(all_frames):
                    frames.append(all_frames[idx])
        else:
            # Fallback to uniform sampling
            indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
            frames = [all_frames[i] for i in indices]
        
        return frames
    
    def save_frames(self, frames, video_name, prefix='frame'):
        """
        Save extracted frames to disk
        
        Args:
            frames (list): List of frames to save
            video_name (str): Name of the video (for organizing saved frames)
            prefix (str): Prefix for frame filenames
            
        Returns:
            list: Paths to saved frames
        """
        video_dir = self.output_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, frame in enumerate(frames):
            frame_path = video_dir / f'{prefix}_{i:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(str(frame_path))
            
        logger.info(f"Saved {len(frames)} frames to {video_dir}")
        return saved_paths
    
    def preprocess_frame(self, frame, target_size=(224, 224)):
        """
        Preprocess a frame for CNN input
        
        Args:
            frame (numpy.ndarray): Input frame
            target_size (tuple): Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def extract_and_preprocess(self, video_path, num_frames=10, target_size=(224, 224)):
        """
        Extract and preprocess frames in one step
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract
            target_size (tuple): Target size for preprocessing
            
        Returns:
            numpy.ndarray: Array of preprocessed frames
        """
        frames = self.extract_frames(video_path, num_frames)
        preprocessed = np.array([self.preprocess_frame(f, target_size) for f in frames])
        
        return preprocessed


if __name__ == "__main__":
    # Example usage
    extractor = FrameExtractor()
    
    # Test with a video file
    # frames = extractor.extract_frames('data/video_1.mp4', num_frames=10)
    # extractor.save_frames(frames, 'video_1')
