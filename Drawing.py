import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class DrawingConfig:
    WINDOW_WIDTH: int = 640
    WINDOW_HEIGHT: int = 480
    COLOR_BAR_HEIGHT: int = 50
    DRAWING_THICKNESS: int = 5
    COLORS: Dict[str, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        self.COLORS = {
            'Green': (0, 255, 0),
            'Red': (0, 0, 255),
            'Blue': (255, 0, 0),
            'Yellow': (0, 255, 255),
            'Black': (0, 0, 0)
        }

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)
    
    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def is_peace_sign(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        return (index_tip.y < middle_tip.y) and (abs(index_tip.x - middle_tip.x) < 0.1)
    
    def is_open_hand(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        return (index_tip.y > middle_tip.y) and (thumb_tip.y > index_tip.y)

class DrawingCanvas:
    def __init__(self, config: DrawingConfig):
        self.config = config
        self.canvas = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
        self.color_selection_area = self._create_color_selection_area()
        self.current_color = config.COLORS['Green']
        
    def _create_color_selection_area(self):
        color_area = np.zeros((self.config.COLOR_BAR_HEIGHT, self.config.WINDOW_WIDTH, 3), dtype=np.uint8)
        color_width = self.config.WINDOW_WIDTH // len(self.config.COLORS)
        for i, color in enumerate(self.config.COLORS.values()):
            cv2.rectangle(color_area, (i * color_width, 0), 
                         ((i + 1) * color_width, self.config.COLOR_BAR_HEIGHT), color, -1)
        return color_area
    
    def update_color_selection(self, x: int):
        color_width = self.config.WINDOW_WIDTH // len(self.config.COLORS)
        color_index = x // color_width
        if color_index < len(self.config.COLORS):
            self.current_color = list(self.config.COLORS.values())[color_index]
            
    def draw_line(self, start_point: Tuple[int, int], end_point: Tuple[int, int]):
        cv2.line(self.canvas, start_point, end_point, 
                 self.current_color, self.config.DRAWING_THICKNESS)
    
    def erase(self, point: Tuple[int, int]):
        cv2.circle(self.canvas, point, self.config.DRAWING_THICKNESS * 2, (0, 0, 0), -1)

class HandDrawingApp:
    def __init__(self):
        self.config = DrawingConfig()
        self.detector = HandGestureDetector()
        self.canvas = DrawingCanvas(self.config)
        self.prev_point = (-1, -1)
        self.drawing = False
        
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            result = self.detector.detect_landmarks(frame)
            
            if result.multi_hand_landmarks:
                self._process_hand_landmarks(frame, result.multi_hand_landmarks[0])
                
            output_frame = self._create_output_frame(frame)
            self._display_instructions(output_frame)
            
            cv2.imshow('Hand Drawing Application', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def _process_hand_landmarks(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        index_tip = hand_landmarks.landmark[self.detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        
        if y < self.config.COLOR_BAR_HEIGHT:
            self.canvas.update_color_selection(x)
        else:
            self._handle_gestures(hand_landmarks, (x, y))
            
        self.detector.draw_landmarks(frame, hand_landmarks)
    
    def _handle_gestures(self, hand_landmarks, current_point):
        if self.detector.is_peace_sign(hand_landmarks):
            self.canvas.update_color_selection(current_point[0])
            self.drawing = False
        elif self.detector.is_open_hand(hand_landmarks):
            self.canvas.erase(current_point)
            self.drawing = False
        else:
            self._handle_drawing(current_point)
    
    def _handle_drawing(self, current_point):
        if self.prev_point != (-1, -1):
            self.canvas.draw_line(self.prev_point, current_point)
        self.prev_point = current_point
        self.drawing = True
    
    def _create_output_frame(self, frame):
        output = cv2.addWeighted(frame, 0.5, self.canvas.canvas, 0.5, 0)
        output[0:self.config.COLOR_BAR_HEIGHT, 0:self.config.WINDOW_WIDTH] = self.canvas.color_selection_area
        return output
    
    def _display_instructions(self, frame):
        instructions = [
            "Pointing Gesture: Draw",
            "Peace Sign: Select Color",
            "Open Hand: Erase",
            "Press 'q' to exit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 70 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
