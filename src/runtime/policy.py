"""Versioned observation/action IPC; World does not import a policy package."""
import json
import os
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

SCHEMA = 'driversi.observation-action.v1'
OBSERVATION_KEYS = ('timestamp', 'log_name', 'ego2global', 'driving_command', 'ego_dynamic_state', 'cams')


def observation(frames, sensor_root, rear_position, heading, rear_to_center):
    # Deliberately exclude annotations, future labels and simulator privileged state.
    rows = []
    for frame in frames:
        row = {k: frame[k] for k in OBSERVATION_KEYS if k != 'cams'}
        row['cams'] = {k: {'data_path': frame['cams'][k]['data_path']}
                       for k in ('CAM_L0', 'CAM_F0', 'CAM_R0')}
        rows.append(row)
    return dict(schema=SCHEMA, frames=rows, sensor_root=str(sensor_root),
                rear_position=list(rear_position), heading=float(heading),
                rear_to_center=float(rear_to_center))


def _array(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate_response(value, request):
    if value.get('schema') != SCHEMA or value.get('request_id') != request['request_id']:
        raise ValueError('Foreign or incompatible policy response')
    if value.get('timestamp_us') != int(request['frames'][-1]['timestamp']):
        raise ValueError('Policy response does not match the current observation')
    trajectory = value.get('trajectory', {})
    result = {}
    for name, shape in [('waypoints',(9,2)),('velocities',(9,2)),('headings',(9,)),('angular_velocities',(9,))]:
        array = np.asarray(trajectory.get(name), dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid native world trajectory: ' + name)
        result[name] = array
    if value.get('dt_seconds') != 0.5 or value.get('frame') != 'world_vehicle_center':
        raise ValueError('Policy time or coordinate convention differs')
    return result


class PolicyProcess:
    def __init__(self, command, *, timeout=600, env=None):
        if not isinstance(command, list) or not command or not all(isinstance(x,str) and x for x in command):
            raise ValueError('Explicit policy argv required')
        executable = Path(command[0])
        if not executable.is_absolute() or not executable.is_file() or 'clones' in executable.resolve().parts:
            raise ValueError('Policy executable must be installed outside clones')
        left, right = socket.socketpair()
        self.channel = Connection(left.detach())
        self.sequence = 0
        self.timeout = timeout
        environment = dict(os.environ if env is None else env)
        environment.pop('PYTHONPATH', None)
        try:
            self.process = subprocess.Popen([*command, '--fd', str(right.fileno())],
                                            pass_fds=(right.fileno(),), env=environment)
        except BaseException:
            self.channel.close()
            raise
        finally:
            right.close()
        try:
            if not self.channel.poll(timeout) or self.channel.recv_bytes(128).decode() != SCHEMA:
                raise TimeoutError('Policy worker did not become ready')
        except BaseException:
            self.close()
            raise

    def predict(self, request):
        self.sequence += 1
        request = {**request, 'request_id': self.sequence}
        self.channel.send_bytes(json.dumps(request, allow_nan=False, default=_array).encode())
        if not self.channel.poll(self.timeout):
            raise TimeoutError('Policy action timeout')
        response = json.loads(self.channel.recv_bytes(1_000_000))
        return validate_response(response, request)

    def close(self):
        self.channel.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
