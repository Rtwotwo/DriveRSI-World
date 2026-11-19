"""SI-unit lane-following IDM evaluated from one shared pre-step snapshot.

NPC routes are initialized from the recorded scenario and map. Speeds and
positions respond to current traffic, including ego; no future logged speed is
replayed. This first backend does not implement lane changes or signal planning.
"""
import numpy as np
from driversi_world.simulation.components.agents.policy.base_policy import BasePolicy
from driversi_world.simulation.common.dataclasses import Trajectory


def idm_acceleration(speed, desired_speed, gap=None, lead_speed=None):
    """All speeds m/s, bumper gap m, acceleration m/s². Comfortable b=2.5."""
    acceleration = 2.0 * (1 - (max(speed, 0) / max(desired_speed, 0.1))**4)
    if gap is not None:
        desired_gap = 2.0 + max(0.0, speed*1.5 + speed*(speed-lead_speed)/(2*np.sqrt(2.0*2.5)))
        acceleration -= 2.0 * (desired_gap / max(gap, 0.1))**2
    return float(np.clip(acceleration, -4.0, 2.0))


def travel(speed, acceleration, seconds):
    """Stop at zero velocity; never integrate a braking NPC backwards."""
    active = min(seconds, speed/-acceleration) if acceleration < 0 else seconds
    return speed*active + 0.5*acceleration*active**2, max(0.0, speed+acceleration*seconds)


class ReactiveIDMPolicy(BasePolicy):
    def __init__(self, agent):
        super().__init__(agent=agent)
        self.desired_speed = max(4.0, min(float(agent.current_speed), 9.7))
        self.nominal_desired_speed = self.desired_speed
        self.last_decision = {}

    @property
    def is_current_step_valid(self):
        return True

    def act(self):
        a = self.agent
        lane = a.navigation.current_lane
        snapshot = self.engine.closed_loop_snapshot
        own = snapshot[a.id]
        longitudinal, lateral = lane.local_coordinates(own['position'])
        leader, gap, lead_speed = None, None, None
        for key, other in snapshot.items():
            if key == a.id or other['length'] <= 0 or other['width'] <= 0:
                continue
            other_s, other_l = lane.local_coordinates(other['position'])
            heading_error = abs(np.arctan2(np.sin(other['heading']-own['heading']),
                                          np.cos(other['heading']-own['heading'])))
            if other_s <= longitudinal or abs(other_l-lateral) > (own['width']+other['width'])/2 + .2 or heading_error > np.pi/3:
                continue
            distance = other_s-longitudinal-(own['length']+other['length'])/2
            if gap is None or distance < gap:
                leader, gap, lead_speed = key, distance, other['speed']
        acceleration = idm_acceleration(own['speed'], self.desired_speed, gap, lead_speed)
        points, headings, velocities = [], [], []
        for t in np.arange(9)*0.5:
            distance, speed = travel(own['speed'], acceleration, t)
            progress = min(longitudinal + distance, lane.length)
            heading = float(lane.heading_theta_at(progress))
            if progress >= lane.length:
                speed = 0.0
            points.append(lane.position(progress, lateral))
            headings.append(heading)
            velocities.append(speed*np.array([np.cos(heading), np.sin(heading)]))
        self.last_decision = dict(leader=leader, gap_m=gap, acceleration=acceleration,
                                  speed_mps=own['speed'], next_speed_mps=float(np.linalg.norm(velocities[1])))
        return Trajectory(waypoints=np.asarray(points), velocities=np.asarray(velocities),
                          headings=np.asarray(headings), angular_velocities=np.gradient(np.unwrap(headings), .5))


class PathStateController:
    """Execute the next state computed by IDM, never the log's next state."""
    def __init__(self, agent):
        self.agent = agent

    def step(self):
        a, t = self.agent, self.agent.trajectory
        a.set_position(t.waypoints[1])
        a.set_velocity(t.velocities[1])
        a.set_heading_theta(t.headings[1])
        a.set_angular_velocity(t.angular_velocities[1])

