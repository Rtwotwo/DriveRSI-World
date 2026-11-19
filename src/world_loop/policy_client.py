"""World-owned client for a resident Policy PPO process, without source imports."""
import json
import os
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path

SCHEMA = 'driversi.ppo.v2'


class PolicyClient:
    def __init__(self, config, timeout=900):
        self.timeout = timeout
        self.sequence = 0
        self.log = (Path(config['output'])/'work/policy.log').open('w')
        left,right = socket.socketpair()
        self.channel = Connection(left.detach())
        environment = dict(os.environ,CUDA_VISIBLE_DEVICES=str(config.get('policy_gpu',1)),PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        environment.pop('PYTHONPATH',None)
        command = [config['policy_python'],'-u','-B','-m','driversi.training.ppo.service',
                   '--fd',str(right.fileno()),'--checkpoint',config['policy_checkpoint'],
                   '--output',config['policy_output'],'--seed',str(config['seed'])]
        # The Policy service validates the frozen anchor manifest and its tensor
        # hashes. World only passes the explicit optional process-level contract;
        # no cached action or behavior probability is inserted into live PPO.
        if config.get('policy_retention_manifest') is not None:
            command.extend(['--retention-manifest',str(config['policy_retention_manifest'])])
        if config.get('policy_training_config') is not None:
            command.extend(['--training-config',str(config['policy_training_config'])])
        self.process = subprocess.Popen(command,pass_fds=(right.fileno(),),env=environment,
            cwd=Path(config['output']),stdout=self.log,stderr=subprocess.STDOUT)
        right.close()
        try:
            if not self.channel.poll(timeout) or self.channel.recv_bytes(128).decode()!=SCHEMA:
                raise RuntimeError('Resident native Policy service did not become ready')
        except BaseException:
            self.close(graceful=False)
            raise

    def call(self, operation, **fields):
        self.sequence += 1
        request = dict(schema=SCHEMA,request_id=self.sequence,op=operation,**fields)
        self.channel.send_bytes(json.dumps(request,allow_nan=False).encode())
        if not self.channel.poll(self.timeout):
            raise TimeoutError(f'Policy operation timed out: {operation}')
        response = json.loads(self.channel.recv_bytes(32_000_000))
        if response.get('schema')!=SCHEMA or response.get('request_id')!=self.sequence:
            raise ValueError('Foreign or stale PPO response')
        if response.get('ok') is not True:
            raise RuntimeError(f"Policy {operation}: {response.get('error')}: {response.get('message')}")
        return response['result']

    def close(self, graceful=True):
        try:
            if graceful and self.process.poll() is None:
                self.call('close')
                self.process.wait(timeout=30)
        finally:
            self.channel.close()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.log.close()

