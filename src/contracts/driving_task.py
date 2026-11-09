"""Explicit geometric driving tasks; no inferred signal or right-of-way rules."""

import numpy as np

from driversi_world.contracts.road import Route

TASK_VERSION = "route-task-v1"

def tasks_from_metadata(metadata, actor_ids):
    value = metadata.get("driving_tasks")
    if value is None:
        return {}
    if (
        not isinstance(value, dict)
        or value.get("version") != TASK_VERSION
        or set(value) != {"version", "actors"}
        or not isinstance(value["actors"], dict)
        or set(value["actors"]) != set(actor_ids)
    ):
        raise ValueError("Driving tasks must cover every controlled actor with a supported version")
    result = {}
    for actor, task in value["actors"].items():
        if (
            not isinstance(task, dict)
            or set(task) != {"route", "endpoint", "corridor_m", "rule_status"}
            or task["endpoint"] not in {"destination", "support_boundary"}
            or task["rule_status"] != "unknown"
        ):
            raise ValueError("Invalid driving task; traffic legality is not established")
        width = task["corridor_m"]
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, float))
            or not np.isfinite(width)
            or not 0.25 <= width <= 4
        ):
            raise ValueError("Task corridor tolerance must be between0.25and4metres")
        route = Route(**task["route"])
        if len(route.points) > 8192:
            raise ValueError("Task route exceeds8192points")
        if task["endpoint"] == "destination" and route.end_reason != "explicit_endpoint":
            raise ValueError("A destination requires an explicitly declared endpoint")
        goals = metadata.get("task_goals_xy", {})
        if actor not in goals or not np.array_equal(goals[actor], route.points[-1]):
            raise ValueError("Task endpoint and world goal differ")
        # Avoid native radius completion on an earlier pass near the endpoint.
        arc = route.distances
        cutoff = max(0.0, arc[-1] - (2.0 + width))
        if cutoff > 0:
            prefix = np.vstack((route.points[arc < cutoff], route.at(cutoff)))
            if len(prefix) >= 2 and Route(prefix).project(route.points[-1])[1] < 2.0 - 1e-6:
                raise ValueError("Route passes within goal radius before final approach")
        result[actor] = (route, task)
    return result
