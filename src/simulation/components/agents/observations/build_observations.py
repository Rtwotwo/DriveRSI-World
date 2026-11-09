# Derived from OpenDriveLab/WorldEngine; modified for DriveRSI owned imports and runtime.
from driversi_world.simulation.components.agents.observations.image_observation import ImageStateObservation
from driversi_world.simulation.components.agents.observations.state_observation import StateObservation


def build_observation(object_id: str):
    """
    Return the navigation class for target object.
    """
    if object_id == 'ego':
        return ImageStateObservation
    else:
        return StateObservation