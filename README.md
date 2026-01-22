# Video Slide Extractor

Automatically extract unique slide images from lecture-style videos by detecting slide transitions and saving representative frames.

## Features

- **Smart Slide Detection**: Uses SSIM (Structural Similarity Index) or histogram comparison to identify when slides change
- **Time-Based Processing**: Specify start/end times to process only portions of a video
- **Configurable Sampling**: Control how frequently frames are checked for changes
- **Timestamp Logging**: Each extracted slide includes its timestamp in the video for easy reference

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- scikit-image (`scikit-image`) - optional, for more accurate SSIM comparison

Install dependencies:
```bash
pip install opencv-python numpy scikit-image
```

## Usage

1. **Configure parameters** in `extract_slides.py`:
   - `VIDEO_PATH`: Path to your video file (e.g., `"./lecture.mp4"`)
   - `OUTPUT_DIR`: Directory where slides will be saved
   - `START_TIME` / `END_TIME`: Time range in seconds (set `None` for full video)
   - `SAMPLE_TIME_INTERVAL`: How often to check for changes (in seconds)
   - `SIMILARITY_THRESHOLD`: Sensitivity for detecting new slides (0-1, lower = more sensitive)
   - `IMAGE_FORMAT`: Output format (`"png"` or `"jpg"`)

2. **Run the script**:
   
   ```bash
   python extract_slides.py
   ```
   
3. **Find extracted slides** in the output directory (default: `./extracted_slides/`)

## Output

Extracted slides are saved with sequential numbering:
- `slide_0001.png`
- `slide_0002.png`
- `slide_0003.png`
- ...

Each slide includes a timestamp in the console output:
```
Saved slide 1: slide_0001.png (time 00:00:00.00)
Saved slide 2: slide_0002.png (time 00:02:15.50, similarity: 0.742)
```
