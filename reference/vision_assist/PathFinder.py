import numpy as np
from heapq import heappop, heappush
from typing import ClassVar, Optional

from vision_assist.models import Grid

class PathFinder:
    """
    PathFinder handles pathfinding operations using the A* algorithm,
    penalising paths with sharp angles across multiple grids for smoother paths.
    """
    _instance: ClassVar[Optional['PathFinder']] = None
    _initialized: bool = False

    def __new__(cls):
        """Ensure only one instance of PathFinder exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the pathfinder only once."""
        if not self._initialized:
            self._initialized = True
            self._grid_lookup: dict[tuple[int, int], Grid] = {}
            self._open_set: list[tuple[float, tuple[int, int]]] = []
            self._closed_set: set[tuple[int, int]] = set()
            self._came_from: dict[tuple[int, int], tuple[int, int]] = {}
            self._g_score: dict[tuple[int, int], float] = {}
            self._f_score: dict[tuple[int, int], float] = {}
            
            self.angle_cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {} # this is not cleared between runs!

    def _reset_path_state(self) -> None:
        """Reset the internal state for a new pathfinding operation."""
        self._open_set.clear()
        self._closed_set.clear()
        self._came_from.clear()
        self._g_score.clear()
        self._f_score.clear()
        # DON'T CLEAR ANGLE CACHE AS IT IS ADVANTAGEOUS TO KEEP
        # self.angle_cache.clear()

    def _heuristic(self, grid1: dict, grid2: dict) -> float:
        """
        Calculate the Manhattan distance between two grids.
        """
        return abs(grid1.coords.x - grid2.coords.x) + \
               abs(grid1.coords.y - grid2.coords.y)

    def _angle_between_grids(self, path: list[tuple[int, int]], segment_size: int) -> float:
        """
        Calculate the angle changes over a sliding window of the path, considering
        both previous and upcoming segments for smoother transitions.

        Args:
            path: List of grid coordinates (x, y) representing the path so far.
            segment_size: Number of grids to consider for angle calculation.

        Returns:
            The maximum angle change across the analyzed segments.
        """
        if len(path) < segment_size:
            return 0

        angles = []
        # Use a sliding window approach to analyze segments
        half = segment_size // 2
        
        for i in range(half, len(path) - half - 1):
            prev_points = path[i - half:i + 1]
            next_points = path[i + 1:i + half + 1]
            
            prev_vector = (
                prev_points[-1][0] - prev_points[0][0],
                prev_points[-1][1] - prev_points[0][1]
            )
            next_vector = (
                next_points[-1][0] - next_points[0][0],
                next_points[-1][1] - next_points[0][1]
            )
            
            # Convert vectors to tuples to use as cache keys
            key = (tuple(prev_vector), tuple(next_vector))
            if key in self.angle_cache:
                angles.append(self.angle_cache[key])
                continue
            
             # Calculate angle between vectors
            dot_product = prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1]
            magnitude_prev = (prev_vector[0]**2 + prev_vector[1]**2)**0.5
            magnitude_next = (next_vector[0]**2 + next_vector[1]**2)**0.5

            if magnitude_prev == 0 or magnitude_next == 0:
                continue

            # Store degrees, not radians: the cache is never cleared between
            # frames, so a cached lookup must match a freshly computed one.
            angle = np.degrees(np.arccos(np.clip(dot_product / (magnitude_prev * magnitude_next), -1.0, 1.0)))
            angles.append(angle)
            self.angle_cache[key] = angle

        return max(angles) if angles else 0

    def _reconstruct_path(self, end_coords: tuple[int, int], start_grid: dict) -> tuple[list[Grid], float]:
        """
        Reconstruct the path from the came_from dictionary.
        """
        path = []
        current = end_coords
        total_cost = self._g_score[current]

        while current in self._came_from:
            path.append(self._grid_lookup[current])
            current = self._came_from[current]

        path.append(start_grid)
        path.reverse()
        return path, total_cost

    def find_path(
        self,
        graph: dict,
        start_grid: Grid,
        end_grid: Grid,
        grid_lookup: dict[tuple[int, int], Grid]
    ) -> tuple[list[Grid], float]:
        """
        Find the optimal path between start and end grids using A* algorithm,
        penalising sharp angle changes for smoother paths.
        """
        self._reset_path_state()
        self._grid_lookup = grid_lookup

        start_coords = (start_grid.coords.x, start_grid.coords.y)
        end_coords = (end_grid.coords.x, end_grid.coords.y)

        # Initialize scores for start position
        self._g_score[start_coords] = 0
        self._f_score[start_coords] = self._heuristic(start_grid, end_grid)
        heappush(self._open_set, (self._f_score[start_coords], start_coords))

        while self._open_set:
            current_coords = heappop(self._open_set)[1]

            if current_coords == end_coords:
                return self._reconstruct_path(end_coords, start_grid)

            self._closed_set.add(current_coords)

            # Retrieve the path so far. This is identical for every neighbour
            # of this node, so it is built once here rather than inside the
            # loop. Rewriting came_from for a neighbour cannot affect it: any
            # ancestor of the current node is already closed, so it is skipped.
            path_so_far = [current_coords]
            previous = current_coords
            while previous in self._came_from:
                previous = self._came_from[previous]
                path_so_far.append(previous)
            path_so_far.reverse()

            # Process neighbours
            for neighbour_coords, distance in graph[current_coords]:
                if neighbour_coords in self._closed_set:
                    continue

                neighbour_grid = self._grid_lookup[neighbour_coords]

                # Calculate angle penalty based on larger segments
                segment_size = 7  # Consider 7 points: 3 before, current, and 3 after
                avg_angle_change = self._angle_between_grids(path_so_far + [neighbour_coords], segment_size)
                # More gradual penalty for angles
                angle_penalty = 0 if avg_angle_change <= 30 else (avg_angle_change / 90) ** 1.5

                # Adjust penalty multiplier to be more lenient
                penalty_multiplier = 1 + (0.5 * (neighbour_grid.penalty or 0)) + angle_penalty * 1.5
                # penalty_multiplier = 1 + (0.5 * (neighbour_grid.penalty or 0))
                tentative_g_score = self._g_score[current_coords] + (distance * penalty_multiplier)

                if (neighbour_coords not in self._g_score or
                        tentative_g_score < self._g_score[neighbour_coords]):
                    self._came_from[neighbour_coords] = current_coords
                    self._g_score[neighbour_coords] = tentative_g_score
                    self._f_score[neighbour_coords] = tentative_g_score + \
                                                      self._heuristic(neighbour_grid, end_grid)

                    if not any(coords == neighbour_coords for _, coords in self._open_set):
                        heappush(self._open_set,
                                 (self._f_score[neighbour_coords], neighbour_coords))

        return [], float('inf')


path_finder = PathFinder()