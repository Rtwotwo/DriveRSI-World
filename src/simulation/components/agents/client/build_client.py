# Derived from OpenDriveLab/WorldEngine; modified for DriveRSI owned imports and runtime.
from driversi_world.simulation.components.agents.client.base_client import BaseClient

def build_client(object_id: str, config):
    """
    Return the client class for target object.
    """
    if object_id == 'ego':
        client = config.get('ego_client')
    else:
        return BaseClient

    if client == 'external_policy_client':
        from driversi_world.simulation.components.agents.client.external_policy_client import ExternalPolicyClient
        return ExternalPolicyClient
    elif client == 'base_client':
        return BaseClient
    else:
        raise NotImplementedError
