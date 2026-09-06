import cv2
import numpy as np
from typing import ClassVar, Optional

from vision_assist.config import grid_size
from vision_assist.models import Corner, PathColours, Path


class PathVisualiser:
    _instance: ClassVar[Optional['PathVisualiser']] = None
    _initialized: bool = False
   
    PATH_COLORS = [
        PathColours(close=(0, 0, 255), mid=(0, 0, 200), far=(0, 0, 150)),  # Blue variants
        PathColours(close=(255, 0, 0), mid=(200, 0, 0), far=(150, 0, 0)),  # Red variants
    ]
   
    def __new__(cls):
        """Ensure only one instance of PathVisualiser exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
   
    def __init__(self):
        """Initialize the path detector only once."""
        if not self._initialized:
            self._initialized = True
            self.frame: np.ndarray | None = None
            self.paths: list[Path] = []
           
    def _draw_path_grid(self, grid: dict, color: tuple[int, int, int]) -> None:
        """Draw a single grid on the frame."""
        x, y = grid.coords.x, grid.coords.y
       
        grid_corners = np.array([
            [x, y],
            [x + grid_size, y],
            [x + grid_size, y + grid_size],
            [x, y + grid_size]
        ], np.int32)
       
        cv2.fillPoly(self.frame, [grid_corners], color)
    
    def _draw_corner_marker(self, section_idx: int, corner: Corner, path: Path) -> None:
        """Draw a corner marker and label at the corner location."""
        cv2.circle(self.frame, (corner.start.x + 10, corner.start.y + 10), 5, (255, 255, 255), -1)
        cv2.circle(self.frame, (corner.end.x + 10, corner.end.y + 10), 5, (255, 255, 255), -1)
        cv2.putText(
            self.frame,
            f"{section_idx + 1} {corner.direction} {corner.shape} {corner.sharpness}",
            (corner.end.x - 100, corner.end.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )   
        
   
    def _draw_path_sections(self, path: Path, path_idx: int) -> None:
        """Draw path sections with measurements and corner detection."""
        if not path.sections:
            return
       
        # Draw grids for each section
        for i, section in enumerate(path.sections):
            # Alternate between blue and red colors for each section
            path_colors = self.PATH_COLORS[i % 2]
            
            # Calculate color based on section position
            section_progress = i / len(path.sections)
            if section_progress < 0.33:
                color = path_colors.far
            elif section_progress < 0.66:
                color = path_colors.mid
            else:
                color = path_colors.close
                
            # Draw section grids
            for grid in section.grids:
                self._draw_path_grid(grid, color)
           
        # Draw connecting lines between sections
        for section in path.sections:
            start_x = section.start.x + (grid_size // 2)
            start_y = section.start.y + (grid_size // 2)
            end_x = section.end.x + (grid_size // 2)
            end_y = section.end.y + (grid_size // 2)
            cv2.line(self.frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Draw corners if detected
        if path.corners:
            for idx, corner in enumerate(path.corners):
                self._draw_corner_marker(idx, corner, path)
           
    def __call__(self, frame: np.ndarray, paths: list[Path]) -> np.ndarray:
        """Process the frame and visualize all paths."""
        self.frame = frame
        self.paths = paths
       
        for idx, path in enumerate(self.paths):
            self._draw_path_sections(path, idx)
               
        return self.frame


# Create singleton for export
path_visualiser = PathVisualiser()