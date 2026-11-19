"""Deterministic bounded trajectory tracking and rear-axle bicycle integration.

The policy samples a full trajectory. This fixed environment transform executes
only the next 0.5 s; PPO probabilities remain those of the original policy action.
"""
import numpy as np


def rotate(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def bicycle_step(state, rear_path, dt=0.5, wheelbase=3.089):
    """Integrate [x,y,yaw,speed,steering] with ten substeps and no reverse motion."""
    value = np.asarray(state, dtype=float).copy()
    path = np.asarray(rear_path, dtype=float)
    if value.shape != (5,) or path.shape != (9, 2) or not np.isfinite(path).all() or not np.isfinite(value).all():
        raise ValueError('Expected finite rear-axle state and current+eight trajectory points')
    target_speed = float(np.clip(np.linalg.norm(path[1] - path[0]) / dt, 0, 12))
    controls = []
    for _ in range(10):
        h = dt / 10
        distances = np.linalg.norm(path - value[:2], axis=1)
        eligible = np.flatnonzero(distances >= max(2.0, value[3] * 0.8))
        target = path[eligible[0] if len(eligible) else -1]
        local = (target - value[:2]) @ rotate(value[2])
        angle = np.arctan2(2 * wheelbase * local[1], max(np.dot(local, local), 1.0))
        desired_steering = float(np.clip(angle, -0.5, 0.5))
        steering_rate = float(np.clip((desired_steering - value[4]) / h, -0.6, 0.6))
        acceleration = float(np.clip(2.0 * (target_speed - value[3]), -4.0, 2.5))
        speed = max(0.0, value[3] + acceleration * h)
        steering = value[4] + steering_rate * h
        mean_speed = (value[3] + speed) / 2
        yaw_rate = mean_speed * np.tan((value[4] + steering) / 2) / wheelbase
        direction = value[2] + yaw_rate * h / 2
        value[:2] += mean_speed * np.array([np.cos(direction), np.sin(direction)]) * h
        value[2] = np.arctan2(np.sin(value[2] + yaw_rate*h), np.cos(value[2] + yaw_rate*h))
        value[3:] = speed, steering
        controls.append((acceleration, steering_rate))
    return value, dict(acceleration=float(np.mean(np.array(controls)[:, 0])),
                       steering_rate=float(np.mean(np.array(controls)[:, 1])), target_speed=target_speed)


class TrajectoryController:
    """Adapt executed bicycle state to WorldEngine's center-based actor storage."""
    def __init__(self, agent):
        self.agent = agent
        self.state = np.r_[agent.rear_vehicle.current_position, agent.current_heading,
                           agent.current_speed, 0.0].astype(float)
        self.control = dict(acceleration=0.0, steering_rate=0.0, target_speed=agent.current_speed)

    def step(self):
        a = self.agent
        headings = np.asarray(a.trajectory.headings)
        centers = np.asarray(a.trajectory.waypoints)
        radius = a.vehicle.rear_axle_to_center
        rear_path = centers - radius * np.column_stack((np.cos(headings), np.sin(headings)))
        previous_yaw_rate = a.current_angular_velocity
        self.state, self.control = bicycle_step(self.state, rear_path, wheelbase=a.vehicle.wheel_base)
        x, y, yaw, speed, steering = self.state
        yaw_rate = speed*np.tan(steering)/a.vehicle.wheel_base
        a.set_position(np.array([x, y]) + rotate(yaw) @ np.array([radius, 0.0]))
        a.set_heading_theta(yaw)
        a.set_velocity(rotate(yaw) @ np.array([speed, yaw_rate*radius]))
        a.set_angular_velocity(yaw_rate)
        a.set_angular_acceleration((yaw_rate - previous_yaw_rate)/0.5)
        a.set_tire_steering(steering)
        a.set_action([steering, self.control['acceleration']])

