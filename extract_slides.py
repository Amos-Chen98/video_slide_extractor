"""
Extract unique slide images from a lecture-style video.
Detects slide transitions and saves one representative image per slide.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Input video path (will try with common extensions if not found)
VIDEO_PATH = "./lecture.mp4"

# Output directory for extracted slides
OUTPUT_DIR = "./extracted_slides"

# Video time range (in seconds): set None to process entire video
START_TIME = 0  # Start time in seconds (e.g., 60 for 1 minute)
END_TIME = 4350    # End time in seconds (e.g., 300 for 5 minutes)

# Sample time interval (process every N seconds, e.g., 1.0 = once per second)
SAMPLE_TIME_INTERVAL = 1.0

# Similarity threshold (0-1): lower = more sensitive to changes
# Frames with similarity below this threshold are considered new slides
SIMILARITY_THRESHOLD = 0.85

# Image format for saved slides
IMAGE_FORMAT = "png"  # or "jpg"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_frame_similarity(frame1, frame2):
    """
    Calculate similarity between two frames using Structural Similarity Index (SSIM).
    Falls back to histogram comparison if SSIM is not available.
    
    Args:
        frame1, frame2: OpenCV images (BGR format)
    
    Returns:
        Similarity score (0-1, where 1 = identical)
    """
    # Resize frames to reduce computation (optional but faster)
    target_size = (640, 480)
    frame1_resized = cv2.resize(frame1, target_size)
    frame2_resized = cv2.resize(frame2, target_size)
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
    
    # Try using SSIM if available (more accurate)
    try:
        from skimage.metrics import structural_similarity as ssim
        score, _ = ssim(gray1, gray2, full=True)
        return score
    except ImportError:
        # Fallback: histogram comparison
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare using correlation (returns value ~0-1)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity


def extract_slides(video_path, output_dir):
    """
    Extract unique slide images from video.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted slides
    
    Returns:
        Number of slides extracted
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame range based on time
    start_frame = int(START_TIME * fps) if START_TIME is not None else 0
    end_frame = int(END_TIME * fps) if END_TIME is not None else total_frames
    sample_frame_interval = int(SAMPLE_TIME_INTERVAL * fps)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    if START_TIME is not None or END_TIME is not None:
        start_time_str = f"{START_TIME}s" if START_TIME is not None else "start"
        end_time_str = f"{END_TIME}s" if END_TIME is not None else "end"
        print(f"Processing range: {start_time_str} to {end_time_str} (frames {start_frame} to {end_frame})")
    print(f"Sampling every {SAMPLE_TIME_INTERVAL}s")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracking variables
    slide_count = 0
    frame_number = 0
    previous_frame = None
    
    # Skip to start frame if specified
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        
        if not ret or frame_number >= end_frame:
            break  # End of video or reached end time
        
        # Only process every Nth frame
        if frame_number % sample_frame_interval == 0:
            
            if previous_frame is None:
                # Save first frame as first slide
                slide_count += 1
                slide_filename = f"slide_{slide_count:04d}.{IMAGE_FORMAT}"
                slide_path = os.path.join(output_dir, slide_filename)
                cv2.imwrite(slide_path, frame)
                timestamp = frame_number / fps
                minutes, seconds = divmod(timestamp, 60)
                hours, minutes = divmod(minutes, 60)
                time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
                print(f"Saved slide {slide_count}: {slide_filename} (time {time_str})")
                previous_frame = frame.copy()
            else:
                # Compare with previous frame
                similarity = calculate_frame_similarity(previous_frame, frame)
                
                # Check if this is a new slide
                if similarity < SIMILARITY_THRESHOLD:
                    slide_count += 1
                    slide_filename = f"slide_{slide_count:04d}.{IMAGE_FORMAT}"
                    slide_path = os.path.join(output_dir, slide_filename)
                    cv2.imwrite(slide_path, frame)
                    timestamp = frame_number / fps
                    minutes, seconds = divmod(timestamp, 60)
                    hours, minutes = divmod(minutes, 60)
                    time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
                    print(f"Saved slide {slide_count}: {slide_filename} (time {time_str}, similarity: {similarity:.3f})")
                    previous_frame = frame.copy()
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    
    return slide_count


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Check if video file exists
        if not os.path.exists(VIDEO_PATH):
            print(f"Error: Video file not found: {VIDEO_PATH}")
            sys.exit(1)
        
        print(f"Found video: {VIDEO_PATH}")
        print()
        
        # Extract slides
        num_slides = extract_slides(VIDEO_PATH, OUTPUT_DIR)
        
        # Print summary
        print()
        print("=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total slides extracted: {num_slides}")
        print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        
        if num_slides == 0:
            print("\nWarning: No slides detected. Try adjusting:")
            print(f"  - SIMILARITY_THRESHOLD (current: {SIMILARITY_THRESHOLD})")
            print(f"  - SAMPLE_TIME_INTERVAL (current: {SAMPLE_TIME_INTERVAL})")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
