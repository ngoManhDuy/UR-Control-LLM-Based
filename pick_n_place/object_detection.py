import numpy as np
import cv2
import torch
import sys
import os

# Fix PyTorch 2.6+ compatibility issue with YOLO model loading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import YOLO with proper PyTorch compatibility
try:
    # For PyTorch 2.6+, we need to handle the weights_only security feature
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    
    # Add safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([DetectionModel])
    
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    raise

from pick_n_place.camera import RealSenseCamera

class ProductDetector:
    def __init__(self, model_path='best.pt'):
        """
        Initialize the product detector
        
        Args:
            model_path (str): Path to the YOLO model weights
        """
        # Initialize YOLO model with PyTorch 2.6+ compatibility
        try:
            # First, try standard loading
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded successfully from {model_path}")
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e) or "__module__" in str(e):
                print(f"PyTorch security issue detected. Attempting alternative loading method...")
                
                # Try with weights_only=False by temporarily patching torch.load
                original_torch_load = torch.load
                
                def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
                    return original_torch_load(f, map_location=map_location, 
                                             pickle_module=pickle_module, 
                                             weights_only=False, **kwargs)
                
                # Temporarily replace torch.load
                torch.load = patched_load
                
                try:
                    self.model = YOLO(model_path)
                    print(f"✓ YOLO model loaded successfully from {model_path} (with weights_only=False)")
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
            else:
                print(f"Error loading YOLO model: {e}")
                print("Make sure the model file 'best.pt' exists in the current directory")
                raise
        
        # Initialize camera
        self.camera = RealSenseCamera()
    
    def connect(self):
        """Connect to the camera"""
        return self.camera.connect()
    
    def disconnect(self):
        """Disconnect from the camera"""
        self.camera.disconnect()
    
    def detect_product(self, confidence_threshold=0.5):
        """
        Detect product in the current camera frame
        
        Args:
            confidence_threshold (float): Minimum confidence score for detection
            
        Returns:
            tuple: (success, bbox, confidence, depth, image, depth_image)
                - success: Boolean indicating if detection was successful
                - bbox: Bounding box [x1, y1, x2, y2]
                - confidence: Detection confidence score
                - depth: Depth value at the center of bbox (in meters)
                - image: Annotated RGB image
                - depth_image: Raw depth image
        """
        # Get frames from RealSense
        color_image, depth_image = self.camera.get_frames()
        if color_image is None or depth_image is None:
            return False, None, None, None, None, None
        
        # Run YOLO detection
        results = self.model(color_image)[0]
        
        # Process results
        best_detection = None
        best_confidence = 0
        
        if results.boxes is not None and len(results.boxes) > 0:
            for detection in results.boxes.data:
                # detection format: [x1, y1, x2, y2, confidence, class_id]
                if detection[5] == 0:  # class_id 0 is "product"
                    confidence = float(detection[4])
                    if confidence > confidence_threshold and confidence > best_confidence:
                        best_detection = detection
                        best_confidence = confidence
        
        if best_detection is None:
            return False, None, None, None, color_image, depth_image
        
        # Get bounding box
        bbox = best_detection[:4].cpu().numpy()
        
        # Get depth at center of bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        # Ensure coordinates are within image bounds
        center_x = max(0, min(center_x, depth_image.shape[1] - 1))
        center_y = max(0, min(center_y, depth_image.shape[0] - 1))
        
        depth_value = depth_image[center_y, center_x] * self.camera.depth_scale
        
        # Draw detection on image
        annotated_image = color_image.copy()
        cv2.rectangle(annotated_image, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Add text with confidence and depth
        text = f"Product: {best_confidence:.2f}, Depth: {depth_value:.3f}m"
        cv2.putText(annotated_image, text, 
                   (int(bbox[0]), int(bbox[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return True, bbox, best_confidence, depth_value, annotated_image, depth_image

def test_detection():
    """Test the product detection"""
    detector = ProductDetector()
    
    if not detector.connect():
        print("Failed to connect to camera")
        return
    
    print("\nStarting product detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Get detection
            success, bbox, confidence, depth, image, depth_image = detector.detect_product()
            
            if image is not None:
                if success:
                    # Add detection information to image
                    info_text = [
                        f"Confidence: {confidence:.3f}",
                        f"Depth: {depth:.3f}m"
                    ]
                    
                    for i, text in enumerate(info_text):
                        cv2.putText(image, text, (10, 50 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show image
                cv2.imshow('Product Detection', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and success:
                    cv2.imwrite('product_detection.jpg', image)
                    print(f"\nDetection saved:")
                    print(f"Confidence: {confidence:.3f}")
                    print(f"Depth: {depth:.3f}m")
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        detector.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detection() 