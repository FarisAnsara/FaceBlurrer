from DetectFace import DetectFace
import cv2

class BlurFace:
    def __init__(self, model_path, conf=0.325, device=0):
        """
        Initialize the BlurFace class.

        Args:
            model_path (str): Path to the YOLO model weights.
            conf (float): Confidence threshold for detection.
            device (int): Device to use for inference (0 for GPU, -1 for CPU).
        """
        self.detector = DetectFace(model_path, conf, device)

    def blur_faces(self, image_path, output_path, blur_strength=(51, 51)):
        """
        Detect and blur faces in an image.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the output image.
            blur_strength (tuple): Kernel size for Gaussian blur.

        Returns:
            bool: True if faces were detected and blurred, False otherwise.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        detections = self.detector.detect_faces(image)
        face_detected = len(detections) > 0
        if not face_detected:
            return False

        for detection in detections:
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            face_region = image[y_min:y_max, x_min:x_max]
            blurred_face = cv2.GaussianBlur(face_region, blur_strength, 0)
            image[y_min:y_max, x_min:x_max] = blurred_face

        cv2.imwrite(output_path, image)
        return True
