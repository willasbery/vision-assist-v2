import pytest

from vision_assist.PathFinder import PathFinder
from vision_assist.models import Coordinate, Grid


@pytest.fixture
def finder():
    pf = PathFinder()
    pf.angle_cache.clear()
    return pf


# An L-shaped path: four cells straight up, then five to the right.
# The turn gives a 90 degree change, so the angle calculation is exercised.
L_PATH = [(0, 0), (0, 20), (0, 40), (0, 60),
          (20, 60), (40, 60), (60, 60), (80, 60), (100, 60)]


def test_angle_is_reported_in_degrees(finder):
    assert finder._angle_between_grids(L_PATH, 7) == pytest.approx(90.0)


def test_angle_is_stable_across_repeated_calls(finder):
    """The angle cache is deliberately never cleared between frames, so a
    cached lookup must return the same units as a freshly computed one."""
    first = finder._angle_between_grids(L_PATH, 7)
    second = finder._angle_between_grids(L_PATH, 7)

    assert first == pytest.approx(second)


def build_field(rows: int, cols: int, blocked: set = frozenset()):
    """Build a rectangular field of grids plus the adjacency graph over it."""
    from collections import defaultdict

    from vision_assist.config import grid_size

    lookup = {}
    grids = []
    for r in range(rows):
        row = []
        for c in range(cols):
            x, y = c * grid_size, r * grid_size
            empty = (r, c) in blocked
            grid = Grid(
                coords=Coordinate(x=x, y=y),
                centre=Coordinate(x=x + grid_size // 2, y=y + grid_size // 2),
                penalty=0.0,
                row=r,
                col=c,
                empty=empty,
                artificial=False,
            )
            row.append(grid)
            if not empty:
                lookup[(x, y)] = grid
        grids.append(row)

    graph = defaultdict(list)
    for (x, y) in lookup:
        for neighbour in ((x + grid_size, y), (x - grid_size, y),
                          (x, y + grid_size), (x, y - grid_size)):
            if neighbour in lookup:
                graph[(x, y)].append((neighbour, float(grid_size)))

    return grids, lookup, graph


def test_find_path_routes_around_an_obstacle(finder):
    """Characterisation test. A wall spans cols 1-3 of row 4, so the route has
    to hug the left edge. Pins the exact output so the optimisation below can
    be shown not to change it."""
    _, lookup, graph = build_field(9, 5, blocked={(4, 1), (4, 2), (4, 3)})

    path, cost = finder.find_path(graph, lookup[(40, 160)], lookup[(0, 0)], lookup)

    assert [(g.coords.x, g.coords.y) for g in path] == [
        (40, 160), (20, 160), (0, 160), (0, 140), (0, 120), (0, 100),
        (0, 80), (0, 60), (0, 40), (0, 20), (0, 0),
    ]
    assert cost == pytest.approx(271.00850556304795)


def test_find_path_returns_no_route_when_unreachable(finder):
    """A wall spanning the full width isolates the target."""
    _, lookup, graph = build_field(9, 3, blocked={(4, 0), (4, 1), (4, 2)})

    path, cost = finder.find_path(graph, lookup[(20, 160)], lookup[(0, 0)], lookup)

    assert path == []
    assert cost == float("inf")
