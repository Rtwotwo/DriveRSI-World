# Derived from OpenDriveLab/WorldEngine; modified for DriveRSI owned imports and runtime.
"""
Definition of block objects, used for managing multiple roads within a block.
"""

import logging
import math
from abc import ABC

import cv2
import numpy as np

from driversi_world.simulation.utils.type import WorldEngineObjectType
from driversi_world.simulation.scenario.scenarios.scenario_description import ScenarioDescription as SD

from driversi_world.simulation.components.maps.blocks.base_block import BaseBlock
from driversi_world.simulation.components.maps.road_networks.edge_road_network import EdgeRoadNetwork
from driversi_world.simulation.components.maps.lanes.scenario_lane import ScenarioLane


class ScenarioBlock(BaseBlock):
    """
    Base class for RoadBlock.
    """

    object_ID = "RoadBlock_"

    def __init__(self, block_index,
                 global_network,
                 random_seed,
                 map_index,
                 map_data,):
        super(ScenarioBlock, self).__init__(block_index, global_network, random_seed)

        self.map_index = map_index
        self.map_data = map_data

        self._sample_topology()
        self.global_network.add(self.block_network, no_intersect=True)

    def _sample_topology(self) -> bool:
        """
        Sample a new topology to fill self.block_network
        """
        # for each map element.
        for object_id, data in self.map_data.items():
            if WorldEngineObjectType.is_lane(data.get('type', False)):
                if len(data[SD.POLYLINE]) <= 1:
                    continue
                lane = ScenarioLane(object_id, self.map_data)
                self.block_network.add_lane(lane)

            elif WorldEngineObjectType.is_sidewalk(data["type"]):
                self.sidewalks[object_id] = {
                    SD.TYPE: WorldEngineObjectType.BOUNDARY_SIDEWALK,
                    SD.POLYGON: np.asarray(data[SD.POLYGON])[..., :2]
                }

            elif WorldEngineObjectType.is_crosswalk(data["type"]):
                self.crosswalks[object_id] = {
                    SD.TYPE: WorldEngineObjectType.CROSSWALK,
                    SD.POLYGON: np.asarray(data[SD.POLYGON])[..., :2]
                }
            else:
                pass
        return True

    @property
    def block_network_type(self):
        return EdgeRoadNetwork



