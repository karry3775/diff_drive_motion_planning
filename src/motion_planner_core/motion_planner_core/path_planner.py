"""Path planner: A* on costmap between waypoints.

If costmap is empty, returns waypoints directly. Otherwise routes
obstacle-free segments between consecutive waypoints using A*.
"""

import heapq
import math
import numpy as np
from motion_planner_core.costmap import Costmap


def plan_path(waypoints: np.ndarray, costmap: Costmap) -> np.ndarray:
    """Plan obstacle-free path through waypoints.

    Args:
        waypoints: (N, 2) array of [x, y] waypoints (assumed obstacle-free).
        costmap: Costmap with obstacles.

    Returns:
        (M, 2) array of obstacle-free path points.
    """
    if waypoints.shape[0] < 2:
        return waypoints
    if costmap.is_empty():
        return waypoints

    all_points = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]

        if _line_is_clear(start, goal, costmap):
            if i == 0:
                all_points.append(start)
            all_points.append(goal)
        else:
            segment = _astar(start, goal, costmap)
            if segment is None:
                # No path found — fall back to direct
                if i == 0:
                    all_points.append(start)
                all_points.append(goal)
            else:
                if i == 0:
                    all_points.extend(segment)
                else:
                    all_points.extend(segment[1:])

    return np.array(all_points)


def _line_is_clear(start: np.ndarray, goal: np.ndarray, costmap: Costmap) -> bool:
    dist = np.linalg.norm(goal - start)
    if dist < 1e-6:
        return True
    n_steps = max(int(dist / costmap.config.resolution), 2)
    for i in range(n_steps + 1):
        t = i / n_steps
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        if not costmap.is_free_world(x, y):
            return False
    return True


def _astar(start: np.ndarray, goal: np.ndarray, costmap: Costmap):
    sr, sc = costmap.world_to_grid(start[0], start[1])
    gr, gc = costmap.world_to_grid(goal[0], goal[1])

    sr, sc = _snap_to_free(sr, sc, costmap)
    gr, gc = _snap_to_free(gr, gc, costmap)
    if sr is None or gr is None:
        return None

    # 8-connected
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    costs = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

    open_set = [(0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()

    while open_set:
        _, cr, cc = heapq.heappop(open_set)
        if (cr, cc) in closed:
            continue
        closed.add((cr, cc))

        if cr == gr and cc == gc:
            path = []
            node = (gr, gc)
            while node in came_from:
                wx, wy = costmap.grid_to_world(node[0], node[1])
                path.append(np.array([wx, wy]))
                node = came_from[node]
            wx, wy = costmap.grid_to_world(sr, sc)
            path.append(np.array([wx, wy]))
            path.reverse()
            return _downsample(path, costmap.config.resolution * 5)

        for (dr, dc), cost in zip(neighbors, costs):
            nr, nc = cr + dr, cc + dc
            if not costmap.in_bounds(nr, nc) or costmap.is_occupied(nr, nc):
                continue
            if (nr, nc) in closed:
                continue
            new_g = g_score[(cr, cc)] + cost
            if new_g < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = new_g
                h = math.sqrt((nr - gr)**2 + (nc - gc)**2)
                heapq.heappush(open_set, (new_g + h, nr, nc))
                came_from[(nr, nc)] = (cr, cc)

    return None


def _snap_to_free(row, col, costmap, max_search=20):
    if costmap.in_bounds(row, col) and not costmap.is_occupied(row, col):
        return row, col
    for r in range(1, max_search):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if costmap.in_bounds(nr, nc) and not costmap.is_occupied(nr, nc):
                    return nr, nc
    return None, None


def _downsample(path, min_dist):
    if len(path) <= 2:
        return path
    result = [path[0]]
    for p in path[1:-1]:
        if np.linalg.norm(p - result[-1]) >= min_dist:
            result.append(p)
    result.append(path[-1])
    return result
