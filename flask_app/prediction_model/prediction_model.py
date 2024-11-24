import cv2
from ultralytics import YOLO

class PredictionModel:
    def __init__(self, model_path='../model/augmented/best.pt'):
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Label dictionary
        self.label_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6',
            33: '7', 34: '8', 35: '9'
        }

    def getPrediction(self, image_path):
        """
        Perform inference using YOLO model on the uploaded image.
        
        Args:
            image_path (str): Path to the uploaded image
            
        Returns:
            dict: Prediction results including detected objects and their confidences
        """
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Failed to load image")

        # Perform prediction
        results = self.model.predict(frame)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            confidences = result.boxes.conf.tolist()
            
            for i in range(len(boxes)):
                if confidences[i] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(classes[i])
                    confidence = confidences[i]
                    
                    detection = {
                        'label': self.label_dict[class_id],
                        'confidence': confidence,
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        }
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections)
        }