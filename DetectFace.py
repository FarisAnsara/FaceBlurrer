import os
from ultralytics import YOLO

class DetectFace:
    def __init__(self, model_path, conf=0.325, device=0):
        """
        Initialize the DetectFace class.

        Args:
            model_path (str): Path to the YOLO model weights.
            conf (float): Confidence threshold for detection.
            device (int): Device to use for inference (0 for GPU, -1 for CPU).
        """
        # if not os.path.exists(model_path):
        #     raise ValueError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Array of detected bounding boxes.
        """
        results = self.model.predict(image, conf=self.conf, device=self.device)
        detections = results[0].boxes.data.cpu().numpy()
        return detections
