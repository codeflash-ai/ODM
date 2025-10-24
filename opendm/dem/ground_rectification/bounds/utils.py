import numpy as np
from scipy.spatial import ConvexHull
from .types import BoxBounds, PolyBounds

def calculate_convex_hull_bounds(points):
    hull = ConvexHull(points)
    return PolyBounds(points[hull.vertices])

def box_from_point_and_size(center, width, height):
    return BoxBounds(center[0] - width / 2, center[0] + width / 2, center[1] - height / 2, center[1] + height / 2)

def box_from_cloud(point_cloud):
    # Reduce to a single pass over the data by using numpy aggregate functions
    xy = point_cloud.get_xy()
    x_col = xy[:, 0]
    y_col = xy[:, 1]
    x_min = np.min(x_col)
    x_max = np.max(x_col)
    y_min = np.min(y_col)
    y_max = np.max(y_col)
    return BoxBounds(x_min, x_max, y_min, y_max)
