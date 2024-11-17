# Hand Drawing Application

# A real-time interactive drawing tool using computer vision and hand gesture 
# recognition. Create digital artwork through your webcam.

## Features

# Hand Gesture Controls:
# - 👆 Pointing Gesture: Draw on the canvas
# - ✌️ Peace Sign: Select colors from the palette
# - 🖐️ Open Hand: Erase drawings

# Color Palette:
# - Green (default)
# - Red
# - Blue
# - Yellow
# - Black

# Interactive Interface:
# - Real-time hand tracking with visualization
# - Color selection bar at the top
# - On-screen usage instructions
# - Smooth and intuitive drawing experience

## Requirements

# Ensure you have the following installed:
$ pip install opencv-python numpy mediapipe

## Installation

# 1. Clone the repository:
$ git clone https://github.com/YourRepo/HandDrawingApp.git

# 2. Navigate to the project directory:
$ cd HandDrawingApp

# 3. Install the required libraries:
$ pip install -r requirements.txt

## Usage

# Run the application:
$ python main.py

# Use the following hand gestures for interaction:
# - Pointing Gesture (Index Finger Up): Draw on the canvas
# - Peace Sign: Move to the color bar to select a color
# - Open Hand: Erase drawings
# - Press 'q' to quit the application

## Technical Details

# - Uses MediaPipe for real-time hand landmark detection.
# - Implements OpenCV for image processing and canvas visualization.
# - Provides smooth line drawing and erasing functionalities.
# - Features a blended canvas overlay with live webcam feed.

## Controls

# Gesture                  Action
# ------------------------ -----------------------
# Pointing (Index Finger)  Draw
# Peace Sign               Select Color
# Open Hand                Erase
# 'q' key                  Exit Application

## Notes

# - Requires a functional webcam.
# - Works best under good lighting conditions.
# - Supports single-hand detection.
# - Mirror effect enabled for an intuitive drawing experience.

## License

# This project is licensed under the MIT License.
