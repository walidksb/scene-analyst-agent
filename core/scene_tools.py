from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_name="yolov8s.pt"):
        self.model = YOLO(model_name)

    def detect(self, image_path: str):
        results = self.model(image_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        names = results[0].names
        detected_objects = [names[int(c)] for c in labels]
        return detected_objects
