import cv2
import numpy as np
from typing import ClassVar, Optional

from vision_assist.config import grid_size
from vision_assist.models import Grid, Peak, ConvexityDefect, Coordinate
from vision_assist.utils import get_closest_grid_to_point, point_to_line_distance


class ProtrusionDetector:
    """
    ProtrusionDetector is a class that detects protrusions in a given image, in the hopes of finding paths.
    """
    _instance: ClassVar[Optional['ProtrusionDetector']] = None
    _initialized: bool = False
    
    def __new__(cls, debug: bool, imshow: bool):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.debug = debug
            cls._instance.imshow = imshow
        return cls._instance
    
    def __init__(self, debug: bool, imshow: bool):
        """Initialize the detector only once."""
        if not self._initialized:
            self._initialized = True
            
            self.frame: np.ndarray | None = None
            self.grids: list[list[Grid]] | None = None
            self.height: int = 0
            self.width: int = 0
            self.binary: np.ndarray | None = None
            
            # For debug image
            self.frames_processed = 0
        
    def _create_binary_image(self) -> np.ndarray:
        binary = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for grid_row in self.grids:
            for grid in grid_row:
                if grid.empty:
                    continue
                
                x, y = grid.coords.to_tuple()
                
                corners = np.array([
                    [x, y],
                    [x + grid_size, y],
                    [x + grid_size, y + grid_size],
                    [x, y + grid_size]
                ], np.int32)
                
                cv2.fillPoly(binary, [corners], 255)
        
        return cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    
    def _find_peak(
        self, 
        defect_centre: Coordinate | None = None, 
        region_around_protrusion: np.ndarray | None = None
    ) -> list[Peak] | None:
        """
        Find the peak of a path and determine its orientation.
        A peak is considered to be pointing upward if it meets certain geometric criteria.
        """
        if region_around_protrusion is None or defect_centre is None:
            region = self.binary
            x_offset, y_offset = 0, 0
        else:
            region = region_around_protrusion
            box_height, box_width = region.shape[:2]
            x_offset = max(0, defect_centre.x - box_width // 2)
            y_offset = max(0, defect_centre.y - box_height // 2)
            
        region_points = np.where(region == 255)
        if not region_points[0].size:
            return []
        
        y_coords, x_coords = region_points
        
        # Find the topmost point(s)
        min_y = np.min(y_coords)
        peak_x_coords = np.sort(x_coords[y_coords == min_y])
        
        if not peak_x_coords.size:
            return []
        
        # Split into groups if there are gaps
        gaps = np.diff(peak_x_coords)
        split_points = np.where(gaps > (grid_size // 4))[0] + 1
        contiguous_groups = np.split(peak_x_coords, split_points)
        
        peaks = []
        
        for group in contiguous_groups:
            # Get the middle x-coordinate of this peak group
            centre_x = int(group[len(group) // 2])
            
            # Find points within a vertical slice below the peak
            slice_width = grid_size
            mask = (x_coords >= centre_x - slice_width//2) & (x_coords <= centre_x + slice_width//2)
            vertical_slice_y = y_coords[mask]
            vertical_slice_x = x_coords[mask]
            
            if len(vertical_slice_y) == 0:
                continue
                
            # Calculate the height and width of the region below the peak
            height = np.max(vertical_slice_y) - min_y
            width = np.max(x_coords) - np.min(x_coords)
            
            # A peak is considered upward-pointing if:
            # 1. The height below the peak is significantly larger than its width
            # 2. The peak is the highest point in its local region
            # 3. There's a continuous vertical path below the peak
            is_upward = (height > width * 0.5 and  # Height should be at least half the width
                        len(vertical_slice_y) > height * 0.5)  # Should have points along the vertical path
            
            orientation = "up" if is_upward else "right" if centre_x > np.mean(x_coords) else "left"
            
            # Add offsets to convert to global coordinates
            centre = Coordinate(x=centre_x + x_offset, y=int(min_y) + y_offset)
            left = Coordinate(x=int(group[0]) + x_offset, y=int(min_y) + y_offset)
            right = Coordinate(x=int(group[-1]) + x_offset, y=int(min_y) + y_offset)
            
            # Debug visualization
            if self.debug and self.imshow and region_around_protrusion is not None:
                debug_img = region.copy()
                # Convert to local coordinates for visualization
                local_centre = Coordinate(x=centre_x, y=int(min_y))
                local_left = Coordinate(x=int(group[0]), y=int(min_y))
                local_right = Coordinate(x=int(group[-1]), y=int(min_y))
                
                cv2.circle(debug_img, local_centre.to_tuple(), 5, 255, -1)
                cv2.circle(debug_img, local_left.to_tuple(), 3, 128, -1)
                cv2.circle(debug_img, local_right.to_tuple(), 3, 128, -1)
                cv2.putText(debug_img, orientation, (local_centre.x - 20, local_centre.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                
                # Draw vertical slice region
                cv2.rectangle(debug_img, 
                            (centre_x - slice_width//2, min_y),
                            (centre_x + slice_width//2, np.max(vertical_slice_y)),
                            128, 1)

                cv2.imshow(f"Peak Analysis - {orientation}", debug_img)
                cv2.waitKey(0)
            
            peaks.append(Peak(
                centre=centre,
                left=left,
                right=right,
                orientation=orientation
            ))
        
        return peaks
        
    def _get_region_around_protrusion(self, point: Coordinate) -> np.ndarray:
        box_height, box_width = self.frame.shape[1] // 4, self.frame.shape[0] // 4
    
        # Calculate crop boundaries in original image
        x_start = max(0, point.x - box_width // 2)
        x_end = min(self.width, point.x + box_width // 2)
        y_start = max(0, point.y - box_height // 2)
        y_end = min(self.height, point.y + box_height // 2)
        
        # Create empty box
        box = np.zeros((box_height, box_width), dtype=np.uint8)
        
        # Get the cropped region from binary image
        binary_cropped = self.binary[y_start:y_end, x_start:x_end]
        
        # Calculate where in the box the cropped region should go
        # If x_start or y_start is 0, it means we're at the image edge
        # and should position the crop at the box edge
        box_x_start = 0 if x_start == 0 else (box_width // 2) - (point.x - x_start)
        box_y_start = 0 if y_start == 0 else (box_height // 2) - (point.y - y_start)
        
        # Calculate end positions
        box_x_end = box_x_start + binary_cropped.shape[1]
        box_y_end = box_y_start + binary_cropped.shape[0]
        
        # Ensure we don't exceed box dimensions
        if box_x_end > box_width:
            binary_cropped = binary_cropped[:, :-(box_x_end - box_width)]
            box_x_end = box_width
        if box_y_end > box_height:
            binary_cropped = binary_cropped[:-(box_y_end - box_height), :]
            box_y_end = box_height
            
        # Place the cropped region in the correct position in the box
        box[box_y_start:box_y_end, box_x_start:box_x_end] = binary_cropped
        
        return box
    
    def _is_valid_bottom_point(self, point: Coordinate) -> bool:
        """Check if a point has a complete column of grids below it."""
        closest_grid = get_closest_grid_to_point(point, self.grids)
        if not closest_grid:
            return False
            
        for row in self.grids[closest_grid.row + 1:]:
            if row[closest_grid.col].empty:
                return False
        return True
    
    def _is_point_near_quadrilateral(self, point: Coordinate, quad: np.ndarray, threshold: int) -> bool:
        """
        Determine if a point is near or inside a quadrilateral.
        
        Args:
            point: Point to check
            quad: List of 4 coordinates forming the quadrilateral
            threshold: Distance threshold
        
        Returns:
            bool: True if point is near or inside the quadrilateral
        """
        # Convert quad to numpy array for calculations
        quad_points = np.array([[p.x, p.y] for p in quad], dtype=np.int32)
        
        # First check if point is inside the quadrilateral
        if cv2.pointPolygonTest(quad_points, point.to_tuple(), False) >= 0:
            return True
            
        # If not inside, calculate distance to each edge
        edge_distances = []
        for i in range(len(quad)):
            next_i = (i + 1) % len(quad)
            
            # Get edge vector
            edge = np.array([quad[next_i].x - quad[i].x, quad[next_i].y - quad[i].y])
            edge_length = np.linalg.norm(edge)
            
            if edge_length == 0:  # Skip if edge has zero length
                continue
                
            # Calculate distance
            dist = point_to_line_distance(point, quad[i], quad[next_i])
            
            # Adjust threshold based on edge orientation
            # Vertical edges (more y difference than x difference) get a larger threshold
            is_vertical = abs(edge[1]) > abs(edge[0])
            adjusted_threshold = threshold * 1.5 if is_vertical else threshold
            
            edge_distances.append((dist, adjusted_threshold))
        
        # Check if point is within adjusted threshold of any edge
        return any(dist < thresh for dist, thresh in edge_distances)
    
    def _get_quadrilateral(self, global_peaks: list[Peak], contour: np.ndarray) -> list[Coordinate]:
        """Get the quadrilateral that encompasses the main path."""
        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_points = np.array([point[0] for point in hull_points])
        
        # Find bottom left point
        left_candidates = hull_points[np.lexsort((hull_points[:, 1], hull_points[:, 0]))]
        left_candidates = [Coordinate(x=int(point[0]), y=int(point[1])) for point in left_candidates]
        
        bottom_left = next(
            (point for point in left_candidates if self._is_valid_bottom_point(point)),
            left_candidates[0]
        )
        
        # Find bottom right point
        right_candidates = hull_points[np.lexsort((hull_points[:, 1], -hull_points[:, 0]))]
        right_candidates = [Coordinate(x=int(point[0]), y=int(point[1])) for point in right_candidates]
        bottom_right = next(
            (point for point in right_candidates if self._is_valid_bottom_point(point)),
            right_candidates[0]
        )
        
        # TODO: CHECK IF THIS OPTIMISATION IS NECESSARY
        # If the left and right are super close, widen the quadrilateral to at least half as wide as the frame
        if abs(bottom_right.x - bottom_left.x) < self.width // 2:
            how_much_to_widen = (self.width // 2) - abs(bottom_right.x - bottom_left.x)
            # find the position across the frame width of each point
            left_ratio = bottom_left.x / (self.width // 2)
            right_ratio = (bottom_right.x - (self.width // 2)) / (self.width // 2)
            
            # if the quadrilateral is skewed to the right, move the right point less than the right
            if right_ratio > left_ratio:
                bottom_right.x = min(self.width, bottom_right.x + how_much_to_widen * 0.4)
                bottom_left.x = max(0, bottom_left.x - how_much_to_widen * 0.6)
            else:
                bottom_right.x = min(self.width, bottom_right.x + how_much_to_widen * 0.6)
                bottom_left.x = max(0, bottom_left.x - how_much_to_widen * 0.4)      
        # END OF OPTIMISATION      
        
        return [
            bottom_left,
            bottom_right,
            max(global_peaks, key=lambda peak: peak.right.x).right,
            min(global_peaks, key=lambda peak: peak.left.x).left
        ]
        
    def _filter_protrusions(
        self, 
        protrusions: list[Coordinate], 
        convex_hull: cv2.Mat, 
        global_peaks: list[Peak],
        distance_threshold: int = 150
    ) -> list[Coordinate]:
        """ 
        Filter out protrusions that are too close to each other.
        """
        if not protrusions:
            return []
        
        def euclidean_distance(p1: Coordinate, p2: Coordinate) -> float:
            return np.linalg.norm(np.array(p1.to_tuple()) - np.array(p2.to_tuple()))
        
        clusters: list[list[Coordinate]] = []
        
        for point in protrusions:
            added_to_cluster = False
            
            # remove any protrusions that are too close to the bottom of the frame
            if point.y > self.height - (self.height // 10):
                continue
            
            for cluster in clusters:
                if any(
                    euclidean_distance(point, cluster_point) < distance_threshold 
                    for cluster_point in cluster
                ):
                    cluster.append(point)
                    added_to_cluster = True
                    break    
            
            if not added_to_cluster:
                clusters.append([point])
                
        # Get the best point from each cluster, by comparing its distance to the convex hull
        filtered_protrusions = []
        
        for cluster in clusters:
            best_point = min(cluster, key=lambda point: cv2.pointPolygonTest(convex_hull, point.to_tuple(), True))
            filtered_protrusions.append(best_point)
            
        for filtered_protrusion in filtered_protrusions:
            for global_peak in global_peaks:
                # add 50% to the threshold to allow for some overlap as we are comparing to the global peak
                if euclidean_distance(filtered_protrusion, global_peak.centre) < distance_threshold * 1.50:
                    filtered_protrusions.remove(filtered_protrusion)
                    break  
            
        return filtered_protrusions      
        
    def _detect_smooth_protrusions(self, contour: np.ndarray) -> list[Coordinate]:
        """Detect protrusions by analyzing contour curvature and direction changes."""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Sample points along the contour at regular intervals
        num_samples = 50
        epsilon = cv2.arcLength(contour, True) * 0.02
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Analyze direction changes
        protrusions = []
        directions = []
        
        for i in range(len(approx_contour)):
            prev_idx = (i - 1) % len(approx_contour)
            next_idx = (i + 1) % len(approx_contour)
            
            # Calculate direction vectors
            prev_vec = approx_contour[i][0] - approx_contour[prev_idx][0]
            next_vec = approx_contour[next_idx][0] - approx_contour[i][0]
            
            # Normalize vectors
            prev_vec = prev_vec / np.linalg.norm(prev_vec)
            next_vec = next_vec / np.linalg.norm(next_vec)
            
            # Calculate direction change
            dot_product = np.dot(prev_vec, next_vec)
            direction_change = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # Check if this is a significant direction change
            if direction_change > np.pi/4:  # 45 degrees
                point = approx_contour[i][0]
                protrusions.append(Coordinate(x=int(point[0]), y=int(point[1])))
                
        return protrusions
    
    def create_debug_image(self) -> np.ndarray:
        # Create RGB debug image
        debug_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw grids first
        for grid_row in self.grids:
            for grid in grid_row:
                x, y = grid.coords.to_tuple()
                corners = np.array([
                    [x, y],
                    [x + grid_size, y],
                    [x + grid_size, y + grid_size],
                    [x, y + grid_size]
                ], np.int32)
                
                # Fill grid with color based on state
                if grid.empty:
                    color = (50, 50, 200)
                elif grid.artificial:
                    color = (50, 200, 50)
                else:
                    color = (200, 50, 50)
                    
                cv2.fillPoly(debug_image, [corners], color)
                
                # Draw grid borders in white
                cv2.polylines(debug_image, [corners], True, (255, 255, 255), 1)
                
        return debug_image
    
    def __call__(self, frame: np.ndarray, grids: list[list[Grid]], grid_lookup: dict[tuple[int, int], Grid]) -> list[Coordinate]:
        """
        Process a new frame to detect protrusions.
        
        Returns:
            list: A list of protrusion coordinates
        """
        self.frame = frame
        self.grids = grids
        self.height, self.width = frame.shape[:2]
        self.binary = self._create_binary_image()
        
        self.frames_processed += 1
        
        # Create debug image (copy of original frame or blank canvas)
        if self.debug:
            debug_image = self.create_debug_image()
        
        # Find global peak - the highest point in the mask, we always kind of assume the user
        # is walking from the bottom to the top of the frame
        global_peaks = self._find_peak()
        if global_peaks is None:
            print("No global peak found.")
            return []
            
        # Process contours
        # contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not contours:
        #     print("No contours found.")
        #     return []
        
        # protrusions = []
        
        # contour = max(contours, key=cv2.contourArea)
        
        # # TODO: fix the smooth protrusions so they do not overfire
        # # smooth_protrusions = self._detect_smooth_protrusions(contour)
        # # protrusions.extend(smooth_protrusions)
            
        # x, y, w, h = cv2.boundingRect(contour)
        
        # # Draw convex hull in green
        # hull = cv2.convexHull(contour)
        # # cv2.drawContours(debug_image, [hull], -1, (0, 255, 0), 2)
        
        # quad = self._get_quadrilateral(global_peaks, contour)
        
        # # Draw quadrilateral in blue
        # quad_points = np.array([[p.x, p.y] for p in quad], dtype=np.int32)
        # # cv2.polylines(debug_image, [quad_points], True, (255, 0, 0), 2)
        
        # # Process defects
        # hull_indices = cv2.convexHull(contour, returnPoints=False)
        # defects = cv2.convexityDefects(contour, hull_indices)
        
        # print("Len defects:", len(defects))
        
        # if defects is None:
        #     return [global_peak.centre for global_peak in global_peaks]
      
        # for defect in defects:
        #     convexity_defect = ConvexityDefect(
        #         start=Coordinate(x=int(contour[defect[0][0]][0][0]), y=int(contour[defect[0][0]][0][1])),
        #         end=Coordinate(x=int(contour[defect[0][1]][0][0]), y=int(contour[defect[0][1]][0][1])),
        #         far=Coordinate(x=int(contour[defect[0][2]][0][0]), y=int(contour[defect[0][2]][0][1])),
        #         depth=float(defect[0][3])
        #     )
                        
        #     if not (convexity_defect.depth > 0.25 * w and 
        #         30 < convexity_defect.angle_degrees < 150 and
        #         convexity_defect.start.y < y + (0.8 * h)):
        #         continue     
                
        #     region_around_protrusion = self._get_region_around_protrusion(convexity_defect.start)            
            
        #     peaks = self._find_peak(convexity_defect.start, region_around_protrusion)
            
        #     for peak in peaks:
        #         point_near_quad = self._is_point_near_quadrilateral(peak.centre, quad, threshold=150)
        #         point_inside = cv2.pointPolygonTest(quad_points, peak.centre.to_tuple(), False) >= 0
                
        #         if not point_near_quad and not point_inside:
        #             protrusions.append(peak.centre)
        #             # cv2.circle(debug_image, (peak.centre.x, peak.centre.y), 5, (0, 0, 255), -1)
                
        # filtered_protrusions = self._filter_protrusions(protrusions, hull, global_peaks)
        
        # DEBUG
        if self.debug and self.imshow:
            # Add legend
            legend_y = 30
            # Draw global peak last so it's on top
            for global_peak in global_peaks:
                cv2.circle(debug_image, (global_peak.centre.x, global_peak.centre.y), 8, (255, 0, 255), -1)  # magenta
                
            cv2.putText(debug_image, "Binary Image: Gray", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.putText(debug_image, "Convex Hull: Green", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_image, "Quadrilateral: Blue", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_image, "Defect Points: Yellow", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_image, "Far Points: Cyan", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_image, "Protrusions: Red", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(debug_image, "Global Peak: Magenta", (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(debug_image, "Frames Processed: " + str(self.frames_processed), (10, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            
            resized_debug = cv2.resize(debug_image, (576, 1024), interpolation=cv2.INTER_AREA)
            
            cv2.imwrite("generate_testing_grids/examples/outputs/grids_visualised_2.png", resized_debug)
            cv2.imshow("Protrusion Detection", resized_debug)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print("Length of global peaks:", len(global_peaks))
            # END OF DEBUG
            
        # return [global_peak.centre for global_peak in global_peaks] + filtered_protrusions
        return [global_peak.centre for global_peak in global_peaks]