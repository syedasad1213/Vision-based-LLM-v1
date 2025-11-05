# week2_object_detection/models/detection_model.py
# YOLOv8 object detection wrapper

from ultralytics import YOLO
from PIL import Image
import numpy as np

class DetectionModel:
    """YOLOv8 object detection model wrapper"""
    
    def __init__(self, model_name="yolov8n.pt", device="cpu"):
        """
        Initialize YOLOv8 model
        
        Args:
            model_name: YOLOv8 model variant (n/s/m/l/x)
            device: 'cuda' or 'cpu'
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.device = device
        print(f"✓ YOLOv8 loaded on {device}")
    
    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run object detection on image
        
        Args:
            image: PIL Image
            conf_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detections with format:
            [
                {
                    'class': 'person',
                    'confidence': 0.95,
                    'bbox': [x1, y1, x2, y2]
                },
                ...
            ]
        """
        # Run inference
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return detections
    
    def detect_and_count(self, image, conf_threshold=0.25):
        """
        Detect objects and return counts by class
        
        Returns:
            dict: {'person': 3, 'car': 2, ...}
        """
        detections = self.detect(image, conf_threshold)
        
        counts = {}
        for det in detections:
            class_name = det['class']
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return counts