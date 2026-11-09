# Derived from OpenDriveLab/WorldEngine; modified for DriveRSI owned imports and runtime.
from driversi_world.simulation.components.agents.controller.tracker.lqr_tracker import LQRTracker


def build_tracker(config):

    tracker = config.get('tracker', 'lqr')
    if tracker == 'lqr':
        return LQRTracker
    else:
        raise NotImplementedError(f'The assigned tracker {tracker} is not'
                                  f'implemented.')
