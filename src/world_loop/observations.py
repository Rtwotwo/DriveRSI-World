"""Current/past camera observations for the versioned Policy process boundary.

Sensor PNGs are temporary IPC artifacts under exp, not a synthetic data export.
Only four actual 2 Hz observations are exposed; privileged geometry stays World-side.
"""
from collections import deque
from pathlib import Path
import numpy as np
from driversi_world.simulation.closed_loop.environment import CAMERAS


def inertial_state(state, control, wheelbase):
    """Rear-axle [vx,vy,ax,ay]; turning requires centripetal acceleration."""
    return [float(state[3]),0.,float(control['acceleration']),
            float(state[3]**2*np.tan(state[4])/wheelbase)]


class ObservationHistory:
    def __init__(self, world, name):
        self.world = world
        self.frames = deque(maxlen=4)
        # An interrupted process may leave valid old PNGs. A distinct run namespace
        # prevents accidentally reusing them when the experiment is restarted.
        run_id = world.config.get('run_id', 'probe')
        if not run_id or Path(run_id).name != run_id:
            raise ValueError('run_id must be a single directory name')
        self.root = world.work/'sensors'/run_id
        self.folder = self.root/name
        self.folder.mkdir(parents=True,exist_ok=False)

    def append(self, rendered):
        import cv2
        world = self.world
        cams = {}
        for camera in CAMERAS:
            image = rendered['cameras'][camera]['image']
            if image.shape != (1080,1920,3):
                raise ValueError('Native Policy adapter requires 1920x1080 cameras')
            path = self.folder/f'{world.step_index:03d}_{camera}.png'
            if not cv2.imwrite(str(path),image):
                raise RuntimeError('Could not write temporary camera observation')
            cams[camera] = {'data_path':path.relative_to(self.root).as_posix()}
        state = world.ego.controller.state
        frame = dict(timestamp=int(world.scene['base_timestamp']+world.step_index*.5*1e6),
            log_name=world.config['scene'],ego2global=rendered['ego2global'].tolist(),
            driving_command=np.asarray(world.raw[world.step_index]['driving_command']).tolist(),
            ego_dynamic_state=inertial_state(state,world.ego.controller.control,world.ego.vehicle.wheel_base),cams=cams)
        if self.frames and frame['timestamp']-self.frames[-1]['timestamp']!=500000:
            raise ValueError('Policy history must advance by exactly 0.5 seconds')
        self.frames.append(frame)

    def payload(self):
        if len(self.frames)!=4:
            raise ValueError('Four actual observations are required before policy sampling')
        state = self.world.ego.controller.state
        return dict(schema='driversi.observation-action.v1',frames=list(self.frames),
            sensor_root=str(self.root),rear_position=state[:2].tolist(),heading=float(state[2]),
            rear_to_center=float(self.world.ego.vehicle.rear_axle_to_center))

    def warmup(self):
        self.append(self.world.reset())
        for _ in range(3):
            observation,_,terminated,truncated,_ = self.world.step(self.world.straight_action())
            if terminated or truncated:
                raise RuntimeError('Warmup ended before four real observations were available')
            self.append(observation)
        return self.payload()
