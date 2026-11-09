"""Run the owned SimEngine with a bounded, qualified DriveRSI configuration."""
import json
import sys
from pathlib import Path

from driversi_world.runtime.paths import configure_runtime, experiment_path


def run(spec):
    output = experiment_path(spec["output"])
    configure_runtime(output)
    # Heavy native imports occur only after all cache/output destinations are fixed.
    import pickle
    from hydra import compose, initialize_config_dir
    from driversi_world.simulation.runner.builders.env_builder import build_envs
    from driversi_world.simulation.runner.executor import run_runners
    from driversi_world.simulation.engine.engine_utils import get_engine, close_engine
    with initialize_config_dir(config_dir=str(Path(__file__).parent / "configs"), version_base="1.2"):
        cfg = compose(config_name="default_runner")
    cfg.output_dir = str(output)
    cfg.data_output_dir = str(output / "openscene_format")
    cfg.data_file_path = spec["scenarios"]
    cfg.asset_folder_path = str(Path(spec["asset_root"]) / "assets")
    cfg.nuplan_map_root = spec["nuplan_maps"]
    cfg.use_planner_actions = True
    cfg.ego_client = "external_policy_client"
    cfg.ego_policy = "env_input_policy"
    cfg.ego_navigation = "trajectory_navigation"
    cfg.ego_controller = "log_play_controller"
    cfg.agent_policy = "idm_policy" if spec["traffic_mode"] == "R" else "trajectory_policy"
    cfg.agent_navigation = "idm_navigation" if spec["traffic_mode"] == "R" else "trajectory_navigation"
    cfg.with_metric_manager = True
    cfg.with_dense_reward_manager = False  # No HydraMDP vocabulary reward substituted for VLA learning.
    cfg.distributed_mode = "SINGLE_NODE"
    cfg.enable_resume = False
    cfg.completed_scenarios_dir = str(output / "completed_scenarios")
    cfg.runner_report_file = "runner_report.json"
    cfg.max_step = spec["max_steps_per_scene"] * len(json.loads(Path(spec["selection"]).read_text())["records"])
    cfg.seed = spec["seed"]
    cfg.planner_data_path = str(output / "plans")
    cfg.planner_client_folder = str(output / "observations")
    cfg.number_of_cpus_allocated_per_simulation = spec["cpu_workers"]
    cfg.policy_command = spec["policy_command"]
    with Path(spec["scenarios"]).open("rb") as handle:
        scenes = pickle.load(handle)
    if not scenes or any(s["log_length"] < cfg.num_history + cfg.num_future for s in scenes.values()):
        raise ValueError("Native NAVSIM window requires four history and eight future frames")
    try:
        run_runners(build_envs(cfg, None, scenes), None, cfg)
    finally:
        engine = get_engine()
        if engine is not None and hasattr(engine,"external_policy"):
            engine.external_policy.close()
        if engine is not None:
            close_engine()


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
