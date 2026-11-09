"""Offline, bounded interpolation of holes in released road-height assets.

The live sampler retains its strict finite bilinear rule. This utility derives a
new asset and never overwrites measured values or uses earlier repairs as data.
A locally planar surface must have support on all sides and pass fixed spatial
holdouts. These tests are geometry evidence, not image or training admission.
"""

def _inside_hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return False
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for sequence, hull in ((points, lower), (reversed(points), upper)):
        for point in sequence:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
    hull = lower[:-1] + upper[:-1]
    return len(hull) >= 3 and all(
        cross(hull[i], hull[(i+1) % len(hull)], (0., 0.)) >= -1e-12
        for i in range(len(hull))
    )
