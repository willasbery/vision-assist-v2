import numpy as np
from typing import ClassVar, Optional

from vision_assist.config import grid_size, penalty_colour_gradient
from vision_assist.models import Coordinate, Grid


class PenaltyCalculator:
    """
    Handles the calculation and management of grid penalties.
    """
    _instance: ClassVar[Optional['PenaltyCalculator']] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Ensure only one instance of PenaltyCalculator exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the penalty calculator only once."""
        if not self._initialized:
            self._initialized = True
            
    def _pre_compute_easy_segments(self, np_grids: np.ndarray, grids: list[list[Grid]]) -> None:
        """
        Pre-compute the easy segments for the np_grids.
        An easy segment is a row or column that is one contiguous segments of 1s.
        The segment doesn't have to be the entire row or column, it can be a subset.
        
        This is used to speed up the penalty calculation.

        Args:
            np_grids (np.ndarray): The numpy array of the grids.
        """
        # easy_rows is a list of tuples, each tuple contains:
        #   - the row index
        #   - the start Coordinate of the segment
        #   - the end Coordinate of the segment
        self.easy_rows: dict[int, tuple[Coordinate, Coordinate]] = {}
        self.easy_cols: dict[int, tuple[Coordinate, Coordinate]] = {}
        
        for row in range(np_grids.shape[0]):
            # Get the indices of the non-empty grids in the row
            indices = np.where(np_grids[row, :] == 1)[0]
            # If the final index minus the first index is equal to the length of 
            # the indices minus 1, then the row is an easy row
            if len(indices) > 0 and indices[-1] - indices[0] == len(indices) - 1:
                self.easy_rows[row] = (grids[row][indices[0]].coords, grids[row][indices[-1]].coords)
                
        for col in range(np_grids.shape[1]):
            indices = np.where(np_grids[:, col] == 1)[0]
            if len(indices) > 0 and indices[-1] - indices[0] == len(indices) - 1:
                self.easy_cols[col] = (grids[indices[0]][col].coords, grids[indices[-1]][col].coords)
    
    def _calculate_segment_penalty(
        self, grid: Grid, grid_lookup: dict[tuple[int, int], Grid], direction: str
    ) -> float:
        """
        Calculate penalty for a grid within a segment (row or column).
        """
        starting_coords = grid.coords
        x, y = starting_coords.x, starting_coords.y
        furthest_left, furthest_right = starting_coords, starting_coords
        
        if direction == "row" and grid.row in self.easy_rows:
            furthest_left, furthest_right = self.easy_rows[grid.row]
        elif direction == "col" and grid.col in self.easy_cols:
            furthest_left, furthest_right = self.easy_cols[grid.col]
        else:
            # Traverse in the left/up direction
            while True:
                next_coords = (
                    (x - grid_size, y) if direction == "row" else (x, y - grid_size)
                )
                if next_coords not in grid_lookup or grid_lookup[next_coords].empty:
                    furthest_left = Coordinate(x=x, y=y)
                    break
                furthest_left = Coordinate(x=next_coords[0], y=next_coords[1])
                x, y = next_coords

            # Reset x, y to the current coordinates
            x, y = starting_coords.x, starting_coords.y

            # Traverse in the right/down direction
            while True:
                next_coords = (
                    (x + grid_size, y) if direction == "row" else (x, y + grid_size)
                )
                if next_coords not in grid_lookup or grid_lookup[next_coords].empty:
                    furthest_right = Coordinate(x=x, y=y)
                    break
                furthest_right = Coordinate(x=next_coords[0], y=next_coords[1])
                x, y = next_coords

        # Calculate the position ratio of the grid within the segment
        denominator = furthest_right.x - furthest_left.x if direction == "row" else furthest_right.y - furthest_left.y
        
        # denominator is 0 when there is only one grid in the segment!! silly edge case was causing a bug
        if denominator == 0:
            position_ratio = 0.5
        else:    
            position_ratio = (
                (starting_coords.x - furthest_left.x) / (denominator)
                if direction == "row"
                else (starting_coords.y - furthest_left.y) / (denominator)
            )
            
        return 2 * abs(position_ratio - 0.5)
    
    def calculate_penalty(self, grid: Grid, grid_lookup: dict[tuple[int, int], Grid]) -> float:
        """
        Calculate penalty for a grid based on its position within continuous segments
        in both its row and column using grid lookup.
        """
        if grid.empty:
            return 0
        
        # if grid.artificial:
        #     return 0.1
    
        # Calculate row and column penalties
        row_penalty = self._calculate_segment_penalty(grid, grid_lookup, "row")
        col_penalty = self._calculate_segment_penalty(grid, grid_lookup, "col")
    
        if row_penalty > 0.99 or col_penalty > 0.99:
            return 1

        total_penalty = row_penalty + col_penalty
        if total_penalty == 0:
            return 0

        # Calculate dominance factor
        dominance_factor = abs(row_penalty - col_penalty) / total_penalty
        row_weight = 0.5 + (0.25 * dominance_factor if row_penalty > col_penalty else -0.25 * dominance_factor)
        col_weight = 1 - row_weight
        
        # Weighted average
        penalty = (row_penalty * row_weight) + (col_penalty * col_weight)        

        return penalty

    def get_penalty_colour(self, penalty: float) -> tuple[int, int, int]:
        """
        Get the colour based on the penalty value.
        """
        closest_key = min(
            penalty_colour_gradient.keys(),
            key=lambda x: abs(x - penalty),
        )
        return penalty_colour_gradient[closest_key]
        # return (255, 255, 255)


penalty_calculator = PenaltyCalculator()
