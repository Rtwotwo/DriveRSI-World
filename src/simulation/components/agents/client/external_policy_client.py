"""Model-independent observed-frame client. No policy source import is permitted."""
from collections import deque
from pathlib import Path
import time

from driversi_world.runtime.files import write_json
from driversi_world.runtime.paths import experiment_path
from driversi_world.runtime.policy import PolicyProcess, observation
from driversi_world.simulation.common.dataclasses import Trajectory
from driversi_world.simulation.components.agents.client.base_client import BaseClient
from driversi_world.simulation.engine.engine_utils import get_engine


class ExternalPolicyClient(BaseClient):
    def __init__(self, agent):
        super().__init__(agent)
        self.engine = get_engine()
        self.config = self.engine.global_config
        self.frames = deque(maxlen=4)

    def process_frame(self, frame_data, step):
        self.frames.append(frame_data)

    def get_trajectory(self, step):
        if step < 3:
            track = self.agent.object_track
            return Trajectory(waypoints=track['position'][step:step+9,:2],
                              velocities=track['velocity'][step:step+9],
                              headings=track['heading'][step:step+9],
                              angular_velocities=track['angular_velocity'][step:step+9])
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        worker = getattr(self.engine, 'external_policy', None)
        if worker is None:
            worker = PolicyProcess(list(self.config.policy_command))
            self.engine.external_policy = worker
        request = observation(list(self.frames), self.config.data_output_dir+'/sensor_blobs',
                              self.agent.rear_vehicle.current_position,
                              self.agent.current_heading, get_pacifica_parameters().rear_axle_to_center)
        started=time.monotonic()
        trajectory=worker.predict(request)
        path=experiment_path(Path(self.config.planner_data_path)/self.engine.current_scene['id']/(str(step)+'.json'))
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,dict(step=step,timestamp_us=int(self.frames[-1]['timestamp']),
                            trajectory={k:v.tolist() for k,v in trajectory.items()},
                            inference_seconds=time.monotonic()-started,trainable=False))
        return Trajectory(**trajectory)
