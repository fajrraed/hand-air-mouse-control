# Hand Air Mouse Control

A gesture-controlled virtual mouse and scrolling system built using Python, OpenCV, MediaPipe, and PyAutoGUI. It captures real-time video feed to track single-hand movements, allowing users to control system cursor navigation, trigger click actions through finger-pinching mechanics, and toggle active scrolling adjustments.

## Features
- Dynamic gesture parsing for cursor tracking and active scrolling modes.
- Integrated linear smoothing filter to eliminate cursor jitter.
- Precise bounding box setup for tailored camera interaction space mapping.
- Instant fallback support reset if tracking boundaries are cleared.

## Requirements
Install the required system dependencies using pip:
```bash
pip install opencv-python mediapipe pyautogui numpy
```
*Note: This project requires the 'hand_landmarker.task' asset file from MediaPipe placed inside the root directory to run hand landmarker tracking.*

## Setup Instructions
1. Download the repository source files to your environment.
2. Download the 'hand_landmarker.task' model file from Google MediaPipe and place it in the project root directory.
3. Run the script:
   ```bash
   python main.py
   ```

## License
This project is open for educational and personal modification purposes.
