"""Executed-state rewards and independent raw metrics, with no candidate scoring."""
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

REWARD_VERSION = 'executed-geometry-v1'


def footprint(position, heading, length, width):
    corners = np.array([[1,1], [1,-1], [-1,-1], [-1,1]]) * [length/2, width/2]
    c, s = np.cos(heading), np.sin(heading)
    return Polygon(corners @ np.array([[c,s],[-s,c]]) + position)


class GeometryReward:
    def __init__(self, scene, ego):
        polygons = []
        for feature in scene['map_features'].values():
            if feature['type'].startswith('LANE') and 'polygon' in feature:
                polygon = Polygon(np.asarray(feature['polygon'])[:, :2]).buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon)
        if not polygons:
            raise ValueError('A real drivable lane polygon map is required')
        self.drivable = unary_union(polygons)
        route = np.asarray(ego.object_track['position'])[:, :2]
        route = route[np.r_[True, np.linalg.norm(np.diff(route,axis=0),axis=1)>1e-3]]
        if len(route)<2:
            raise ValueError('Scene has no usable route for progress measurement')
        tangent = route[-1]-route[-2]
        route = np.vstack((route,route[-1]+30*tangent/np.linalg.norm(tangent)))
        self.route = LineString(route)
        self.progress = self.route.project(Point(ego.current_position))

    def evaluate(self, ego, participants, control):
        shape = footprint(ego.current_position, ego.current_heading, ego.length, ego.width)
        offroad = float(np.clip(1-self.drivable.intersection(shape).area/shape.area,0,1))
        collisions = []
        nearest = float('inf')
        for key, actor in participants.items():
            if key=='ego' or actor['length']<=0 or actor['width']<=0:
                continue
            other = footprint(actor['position'],actor['heading'],actor['length'],actor['width'])
            nearest = min(nearest,shape.distance(other))
            if shape.intersection(other).area>1e-4:
                collisions.append(key)
        point = Point(ego.current_position)
        progress = self.route.project(point)
        delta = progress-self.progress
        self.progress = progress
        lateral = self.route.distance(point)
        terms = dict(progress=.2*delta, lateral=-.05*lateral**2,
                     acceleration=-.02*control['acceleration']**2,
                     steering_rate=-.01*control['steering_rate']**2,
                     offroad=-2*offroad, collision=-10*bool(collisions))
        metrics = dict(progress_m=float(progress),delta_progress_m=float(delta),
                       lateral_m=float(lateral),offroad_fraction=offroad,collisions=collisions,
                       nearest_clearance_m=None if not np.isfinite(nearest) else float(nearest))
        return float(sum(terms.values())), terms, metrics, bool(collisions or offroad>.2)
