import cv2
from pathlib import Path
import time

class MockCamera:
    """
    A class that simulates a real-time camera feed using a video file.
    It controls the frame rate to mimic real-time capture behavior.
    """
    def __init__(self, video_path: str | Path, target_fps: float = None):
        """
        Initialize the mock camera with a video file.
        
        Args:
            video_path: Path to the video file
            target_fps: Desired frames per second (if None, uses video's original FPS)
        """
        self.cap = cv2.VideoCapture(str(video_path))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps if target_fps is not None else self.original_fps
        self.frame_delay = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        # Store video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def read(self):
        """
        Read a frame from the mock camera, simulating real-time capture.
        
        Returns:
            tuple: (success, frame) similar to cv2.VideoCapture.read()
        """
        # Enforce frame rate
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time
        
        if time_since_last_frame < self.frame_delay:
            time.sleep(self.frame_delay - time_since_last_frame)
            
        ret, frame = self.cap.read()
        
        # Loop video if we've reached the end
        # if not ret:
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     ret, frame = self.cap.read()
            
        self.last_frame_time = time.time()
        return ret, frame
    
    def get(self, propId):
        """
        Get camera property, mimicking cv2.VideoCapture.get()
        
        Args:
            propId: Property identifier (cv2.CAP_PROP_*)
            
        Returns:
            Value of the property
        """
        return self.cap.get(propId)
    
    def isOpened(self):
        """Check if the mock camera is initialized properly."""
        return self.cap.isOpened()
    
    def release(self):
        """Release the video capture object."""
        self.cap.release()