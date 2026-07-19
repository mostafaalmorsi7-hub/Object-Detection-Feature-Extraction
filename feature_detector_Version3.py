import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    contour: np.ndarray
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    rect: Tuple[int, int, int, int]
    circularity: float

class ObjectDetector:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.detections: List[Detection] = []
    
    def find_contours(self, threshold: int = 127, method: str = 'binary') -> List[np.ndarray]:
        if method == 'adaptive':
            binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def analyze(self, contours: List[np.ndarray], min_area: float = 100) -> List[Detection]:
        self.detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            M = cv2.moments(contour)
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
            x, y, w, h = cv2.boundingRect(contour)
            self.detections.append(Detection(contour, area, perimeter, (cx, cy), (x, y, w, h), circularity))
        return self.detections
    
    def detect_circles(self) -> np.ndarray:
        return cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
    
    def detect_lines(self) -> np.ndarray:
        edges = cv2.Canny(self.gray, 100, 200)
        return cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    def draw(self, color=(0, 255, 0), thickness=2) -> np.ndarray:
        result = self.image.copy()
        for det in self.detections:
            x, y, w, h = det.rect
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(result, det.centroid, 5, (0, 0, 255), -1)
        return result
    
    def filter(self, min_area=0, max_area=float('inf'), min_circ=0) -> List[Detection]:
        return [d for d in self.detections if min_area <= d.area <= max_area and d.circularity >= min_circ]