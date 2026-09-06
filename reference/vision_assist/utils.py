import numpy as np

from vision_assist.models import Coordinate, Grid


def get_closest_grid_to_point(point: Coordinate, grids: list[list[Grid]]):
    """
    Get the closest grid to a point.
    
    Args:
        point: The point to find the closest grid to
        grids: The list of grids
    
    Returns:
        The closest grid to the point    
    """    
    closest_grid: dict = None
    min_distance = np.inf
    
    for grid_row in grids:
        for grid in grid_row:
            if grid.empty:
                continue
            
            # Calculate euclidean distance
            distance = np.sqrt((point.x - grid.centre.x) ** 2 + (point.y - grid.centre.y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_grid = grid
    
    return closest_grid


def point_to_line_distance(point: Coordinate, line_start: Coordinate, line_end: Coordinate) -> float:
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    Args:
        point: The point
        line_start: The start of the line segment
        line_end: The end of the line segment
    
    Returns:
        The perpendicular distance from the point to the line segment
    """
    x, y = point.to_tuple()
    x1, y1 = line_start.to_tuple()
    x2, y2 = line_end.to_tuple()
    
    # Calculate the distance
    numerator = abs((y2 -y1 ) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    if denominator == 0:
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    
    return numerator / denominator