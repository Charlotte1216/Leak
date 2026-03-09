"""
train.py
Multi-agent training entry point for gas leak localization.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from marl_leakage_search.agents.marl_agent import MARLAgent
from marl_leakage_search.agents.marl_trainer import MARLTrainer
from marl_leakage_search.envs.marl_env import PlumeEnv


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "marl_leakage_search").exists():
            return parent
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
EXPERIMENT_LOG_DIR = REPO_ROOT / "marl_leakage_search" / "experiments" / "Train_network"


DEFAULT_CONFIG: Dict[str, Any] = {
    "training": {
        "num_episodes": 200,
        "max_steps_per_episode": 500,
        "save_interval": 50,
        "log_interval": 10,
        "pretrained_dir": "",
    },
    "marl": {
        "algorithm": "mappo",  # "mappo" or "qmix"
        "mappo": {"gamma": 0.99, "gae_lambda": 0.95},
        "qmix": {"gamma": 0.99, "mixer_lr": 3e-4, "tau": 0.005},
    },
    "environment": {
        "num_agents": 2,
        "action_dim": 8,
        "source_find_radius": 2.0,
        "collision_penalty": 1.0,
        "battery_penalty": 0.01,
        "found_source_bonus": 5.0,
        "done_bonus": 20.0,
        "success_done_bonus": None,
        "failure_done_penalty": 0.0,
        "enable_collision": True,
        "stop_on_collision": True,
        "observe_wind": False,
        "observe_velocity": False,
        "distance_reward_scale": 0.0,
        "found_source_concentration_scale": 1.0,
        "found_source_stay_penalty": 0.0,
        "found_source_stay_radius_scale": 1.0,
        "uav_params": {},
        "init_pos_mode": "random",
        "dynamic_field": {
            "enabled": False,
            "dt": 1.0,
            "keep_plume_behind_obstacle": True,
            "wind": {
                "enabled": False,
                "speed_amplitude": 0.0,
                "speed_frequency": 0.0,
                "speed_phase": 0.0,
                "dir_amplitude": 0.0,
                "dir_frequency": 0.0,
                "dir_phase": 0.0,
            },
            "vortex": {
                "use_strouhal": False,
                "strouhal": 0.2,
            },
        },
    },
    "agent": {
        "algorithm": "ppo",  # "ppo" or "dqn"
        "network_type": "ffnn",  # "ffnn", "lstm", "transformer"
        "learning": {
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "dqn": {
                "epsilon": 1.0,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.995,
                "tau": 0.005,
                "replay_buffer_size": 10000,
            },
            "ppo": {
                "clip": 0.2,
                "epochs": 4,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "aux": {
                    "enabled": False,
                    "weight": 0.0,
                    "target_index": 2,
                    "hidden_dim": 128,
                },
                "lstm_training": {
                    "enabled": False,
                    "seq_len": 16,
                    "stride": 4,
                },
            },
        },
        "device": "cuda",
    },
    "output": {
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
    },
    "seed": 42,
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_agent_config(agent_cfg: Dict[str, Any]) -> Dict[str, Any]:
    learning = agent_cfg.get("learning", {})
    dqn_cfg = learning.get("dqn", {})
    ppo_cfg = learning.get("ppo", {})
    aux_cfg = ppo_cfg.get("aux", {})
    lstm_training_cfg = ppo_cfg.get("lstm_training", {})
    network_cfg = agent_cfg.get("network", {})
    lstm_net_cfg = network_cfg.get("lstm", {})

    return {
        "lr": float(learning.get("lr", 3e-4)),
        "gamma": learning.get("gamma", 0.99),
        "batch_size": learning.get("batch_size", 64),
        "epsilon": dqn_cfg.get("epsilon", 1.0),
        "epsilon_min": dqn_cfg.get("epsilon_min", 0.01),
        "epsilon_decay": dqn_cfg.get("epsilon_decay", 0.995),
        "tau": dqn_cfg.get("tau", 0.005),
        "replay_buffer_size": dqn_cfg.get("replay_buffer_size", 10000),
        "ppo_clip": ppo_cfg.get("clip", 0.2),
        "ppo_epochs": ppo_cfg.get("epochs", 4),
        "value_coef": ppo_cfg.get("value_coef", 0.5),
        "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
        "aux_enabled": bool(aux_cfg.get("enabled", False)),
        "aux_weight": float(aux_cfg.get("weight", 0.0)),
        "aux_target_index": int(aux_cfg.get("target_index", 2)),
        "aux_hidden_dim": int(aux_cfg.get("hidden_dim", 128)),
        "lstm_hidden_dim": int(lstm_net_cfg.get("hidden_dim", 128)),
        "lstm_lstm_hidden_dim": int(lstm_net_cfg.get("lstm_hidden_dim", 64)),
        "lstm_num_layers": int(lstm_net_cfg.get("num_layers", 1)),
        "ppo_lstm_seq_training_enabled": bool(lstm_training_cfg.get("enabled", False)),
        "ppo_lstm_seq_len": int(lstm_training_cfg.get("seq_len", 16)),
        "ppo_lstm_seq_stride": int(lstm_training_cfg.get("stride", 4)),
        "device": agent_cfg.get("device", "cuda"),
    }


def _save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _snapshot_run_artifacts(
    *,
    config: Dict[str, Any],
    output_cfg: Dict[str, Any],
    file_tag: str,
    train_config_path: str,
    agent_config_path: str,
) -> None:
    """Persist per-run configs so each training run is reproducible and not overwritten."""
    base_log_dir = Path(output_cfg["log_dir"])
    base_save_dir = Path(output_cfg["save_dir"])
    run_log_dir = base_log_dir / file_tag
    run_save_dir = base_save_dir / file_tag
    run_artifact_dir = run_save_dir / "run_artifacts"

    base_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Keep legacy path for analysis scripts and add per-run config snapshot.
    _save_json(config, base_log_dir / "config.json")
    _save_json(config, run_log_dir / "config.json")
    _save_json(config, run_artifact_dir / "merged_config.json")

    train_cfg_src = Path(train_config_path)
    if train_cfg_src.exists():
        shutil.copy2(train_cfg_src, run_artifact_dir / "train_config_input.yaml")

    agent_cfg_src = Path(agent_config_path)
    if agent_cfg_src.exists():
        shutil.copy2(agent_cfg_src, run_artifact_dir / "agent_config_input.yaml")

    _save_json(
        {
            "file_tag": file_tag,
            "train_config_path": str(train_cfg_src.resolve()),
            "agent_config_path": str(agent_cfg_src.resolve()),
            "run_log_dir": str(run_log_dir.resolve()),
            "run_checkpoint_dir": str(run_save_dir.resolve()),
            "aux_enabled": bool(
                config.get("agent", {})
                .get("learning", {})
                .get("ppo", {})
                .get("aux", {})
                .get("enabled", False)
            ),
        },
        run_artifact_dir / "run_meta.json",
    )


def _setup_logging(log_dir: Path, log_file: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("marl_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def _append_avg_reward(
    csv_path: Path,
    episode: int,
    avg_reward: float,
    avg_found_sources: float,
    avg_found_ratio: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0] if rows else []
        if "avg_found_ratio" not in header:
            migrated_rows = [["episode", "avg_reward", "avg_found_sources", "avg_found_ratio"]]
            for row in rows[1:]:
                if not row:
                    continue
                episode_val = row[0] if len(row) > 0 else ""
                avg_reward_val = row[1] if len(row) > 1 else ""
                avg_found_sources_val = row[2] if len(row) > 2 else ""
                avg_found_ratio_val = row[3] if len(row) > 3 else ""
                migrated_rows.append(
                    [episode_val, avg_reward_val, avg_found_sources_val, avg_found_ratio_val]
                )
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(migrated_rows)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "avg_reward", "avg_found_sources", "avg_found_ratio"])
        writer.writerow(
            [
                episode,
                f"{avg_reward:.6f}",
                f"{avg_found_sources:.4f}",
                f"{avg_found_ratio:.6f}",
            ]
        )

def train_loop(
    trainer: MARLTrainer,
    env: PlumeEnv,
    cfg: Dict[str, Any],
    logger: logging.Logger,
    avg_reward_csv: Path,
    checkpoint_run_tag: str | None = None,
    log_run_tag: str | None = None,
) -> None:
    training_cfg = cfg["training"]
    num_episodes = int(training_cfg["num_episodes"])
    max_steps = int(training_cfg["max_steps_per_episode"])
    save_interval = int(training_cfg["save_interval"])
    log_interval = int(training_cfg["log_interval"])

    output_cfg = cfg["output"]
    save_dir = Path(output_cfg["save_dir"])
    if checkpoint_run_tag:
        # Group checkpoints by run hyperparameters so runs do not overwrite each other.
        save_dir = save_dir / checkpoint_run_tag
    base_log_dir = Path(output_cfg["log_dir"])
    run_log_dir = base_log_dir / log_run_tag if log_run_tag else base_log_dir
    run_log_dir.mkdir(parents=True, exist_ok=True)

    trainer.training_stats.setdefault("found_sources", [])
    trainer.training_stats.setdefault("total_sources", [])
    trainer.training_stats.setdefault("found_source_ratios", [])
    trainer.training_stats.setdefault("success_episodes", [])
    trainer.training_stats.setdefault("partial_success_episodes", [])
    trainer.training_stats.setdefault(
        "reward_components",
        {
            "collision": [],
            "concentration": [],
            "found_bonus": [],
            "battery": [],
            "distance": [],
            "found_source_stay": [],
            "success_done": [],
            "failure_done": [],
        },
    )

    for episode in range(num_episodes):
        observations = env.reset()

        for agent in trainer.agents:
            agent.reset()
            agent.train_mode()

        if trainer.algorithm == "mappo":
            trajectories = [dict(states=[], next_states=[], actions=[], rewards=[], dones=[], old_log_probs=[], values=[])
                            for _ in range(trainer.num_agents)]
        else:
            batch_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "global_states": [],
                "next_global_states": [],
            }

        episode_reward = 0.0
        episode_length = 0
        component_sums = {
            "collision": 0.0,
            "concentration": 0.0,
            "found_bonus": 0.0,
            "battery": 0.0,
            "distance": 0.0,
            "found_source_stay": 0.0,
            "success_done": 0.0,
            "failure_done": 0.0,
        }

        for _step in range(max_steps):
            actions = []
            action_log_probs = []
            critic_values = []

            for agent_idx, agent in enumerate(trainer.agents):
                obs = observations[agent_idx]
                if trainer.algorithm == "mappo":
                    action, log_prob, value = agent.select_action_with_probs(obs)
                    actions.append(action)
                    action_log_probs.append(float(log_prob.detach().cpu().item()))
                    critic_values.append(float(value.detach().cpu().item()))
                else:
                    action = agent.select_action(obs, training=True)
                    actions.append(action)

            next_observations, rewards, done, info = env.step(actions)
            done_list = [done for _ in range(trainer.num_agents)]

            if info and "reward_components" in info:
                comps = info["reward_components"]
                for key in component_sums:
                    component_values = comps.get(key)
                    if component_values is None:
                        continue
                    if len(component_values) > 0:
                        component_sums[key] += float(np.mean(component_values))

            if trainer.algorithm == "mappo":
                for agent_idx in range(trainer.num_agents):
                    trajectories[agent_idx]["states"].append(observations[agent_idx])
                    trajectories[agent_idx]["next_states"].append(next_observations[agent_idx])
                    trajectories[agent_idx]["actions"].append(actions[agent_idx])
                    trajectories[agent_idx]["rewards"].append(rewards[agent_idx])
                    trajectories[agent_idx]["dones"].append(done_list[agent_idx])
                    trajectories[agent_idx]["old_log_probs"].append(action_log_probs[agent_idx])
                    trajectories[agent_idx]["values"].append(critic_values[agent_idx])
            else:
                global_state = trainer._get_global_state(observations)
                next_global_state = trainer._get_global_state(next_observations)

                batch_data["states"].append(observations)
                batch_data["actions"].append(actions)
                batch_data["rewards"].append(float(np.mean(rewards)))
                batch_data["next_states"].append(next_observations)
                batch_data["dones"].append(bool(done))
                batch_data["global_states"].append(global_state)
                batch_data["next_global_states"].append(next_global_state)

            episode_reward += float(np.mean(rewards))
            episode_length += 1
            observations = next_observations

            if done:
                break

        if trainer.algorithm == "mappo":
            stats = trainer.train_step_mappo(trajectories)
        else:
            batch = trainer._prepare_qmix_batch(batch_data)
            stats = trainer.train_step_qmix(batch)

        episode_found_sources = int(env.found_sources.sum()) if env.found_sources is not None else 0
        total_sources = int(len(env.sources)) if env.sources is not None else 0
        episode_found_ratio = (
            float(episode_found_sources) / float(total_sources) if total_sources > 0 else 0.0
        )
        episode_success = 1 if (total_sources > 0 and episode_found_sources >= total_sources) else 0
        episode_partial_success = 1 if episode_found_sources > 0 else 0
        trainer.training_stats["episode_rewards"].append(episode_reward)
        trainer.training_stats["episode_lengths"].append(episode_length)
        trainer.training_stats["losses"].append(stats.get("loss", 0.0))
        trainer.training_stats["found_sources"].append(episode_found_sources)
        trainer.training_stats["total_sources"].append(total_sources)
        trainer.training_stats["found_source_ratios"].append(episode_found_ratio)
        trainer.training_stats["success_episodes"].append(episode_success)
        trainer.training_stats["partial_success_episodes"].append(episode_partial_success)
        if episode_length > 0:
            for key in component_sums:
                trainer.training_stats["reward_components"][key].append(component_sums[key] / episode_length)
        else:
            for key in component_sums:
                trainer.training_stats["reward_components"][key].append(0.0)

        if (episode + 1) % log_interval == 0:
            avg_reward = float(np.mean(trainer.training_stats["episode_rewards"][-log_interval:]))
            avg_length = float(np.mean(trainer.training_stats["episode_lengths"][-log_interval:]))
            avg_loss = float(np.mean(trainer.training_stats["losses"][-log_interval:]))
            avg_found_sources = float(np.mean(trainer.training_stats["found_sources"][-log_interval:]))
            avg_found_ratio = float(np.mean(trainer.training_stats["found_source_ratios"][-log_interval:]))
            avg_success_rate = float(np.mean(trainer.training_stats["success_episodes"][-log_interval:]))
            avg_partial_success_rate = float(np.mean(trainer.training_stats["partial_success_episodes"][-log_interval:]))
            avg_components = {
                key: float(np.mean(trainer.training_stats["reward_components"][key][-log_interval:]))
                for key in component_sums
            }
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.2f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Avg Found Sources: {avg_found_sources:.2f} | "
                f"Avg Found Ratio: {avg_found_ratio:.2%} | "
                f"Success Rate: {avg_success_rate:.2%} | "
                f"Partial Success Rate: {avg_partial_success_rate:.2%} | "
                f"Avg Reward Components: "
                f"conc={avg_components['concentration']:.2f}, "
                f"found={avg_components['found_bonus']:.2f}, "
                f"collision={avg_components['collision']:.2f}, "
                f"battery={avg_components['battery']:.2f}, "
                f"distance={avg_components['distance']:.2f}, "
                f"stay_found={avg_components['found_source_stay']:.2f}, "
                f"success_done={avg_components['success_done']:.2f}, "
                f"failure_done={avg_components['failure_done']:.2f}"
            )
            _append_avg_reward(
                avg_reward_csv,
                episode + 1,
                avg_reward,
                avg_found_sources,
                avg_found_ratio,
            )

        if (episode + 1) % save_interval == 0:
            checkpoint_dir = save_dir / f"episode_{episode + 1}"
            trainer.save_models(str(checkpoint_dir))
            _save_json(trainer.training_stats, run_log_dir / "training_stats.json")
            if run_log_dir != base_log_dir:
                _save_json(trainer.training_stats, base_log_dir / "training_stats.json")

    trainer.save_models(str(save_dir / "final"))
    _save_json(trainer.training_stats, run_log_dir / "training_stats.json")
    if run_log_dir != base_log_dir:
        _save_json(trainer.training_stats, base_log_dir / "training_stats.json")


def main() -> None:
    default_field_dir = REPO_ROOT / "marl_leakage_search" / "envs" / "generated_fields"

    parser = argparse.ArgumentParser(description="Train MARL agents for gas leak localization.")
    parser.add_argument("--train-config", type=str, default=str(REPO_ROOT / "marl_leakage_search" / "configs" / "train_config.yaml"))
    parser.add_argument("--agent-config", type=str, default=str(REPO_ROOT / "marl_leakage_search" / "configs" / "agent_config.yaml"))
    parser.add_argument("--field-dir", type=str, default=str(default_field_dir))
    parser.add_argument("--pretrained-dir", type=str, default=None, help="Path to a checkpoint directory to load before training")
    args = parser.parse_args()

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config = _deep_update(config, _load_yaml(args.train_config))
    config = _deep_update(config, {"agent": _load_yaml(args.agent_config)})

    seed = int(config.get("seed", 42))
    _set_seed(seed)

    env_cfg = config["environment"]
    uav_params = env_cfg.get("uav_params", {})
    if not isinstance(uav_params, dict):
        raise TypeError("environment.uav_params must be a mapping/dict in train_config.yaml")

    env = PlumeEnv(
        field_dir=args.field_dir,
        num_agents=int(env_cfg.get("num_agents", 2)),
        source_find_radius=float(env_cfg.get("source_find_radius", 2.0)),
        collision_penalty=float(env_cfg.get("collision_penalty", 1.0)),
        battery_penalty=float(env_cfg.get("battery_penalty", 0.01)),
        found_source_bonus=float(env_cfg.get("found_source_bonus", 5.0)),
        done_bonus=float(env_cfg.get("done_bonus", 20.0)),
        success_done_bonus=(
            None
            if env_cfg.get("success_done_bonus", None) is None
            else float(env_cfg.get("success_done_bonus"))
        ),
        failure_done_penalty=float(env_cfg.get("failure_done_penalty", 0.0)),
        enable_collision=bool(env_cfg.get("enable_collision", True)),
        stop_on_collision=bool(env_cfg.get("stop_on_collision", True)),
        observe_wind=bool(env_cfg.get("observe_wind", False)),
        observe_velocity=bool(env_cfg.get("observe_velocity", False)),
        distance_reward_scale=float(env_cfg.get("distance_reward_scale", 0.0)),
        found_source_concentration_scale=float(env_cfg.get("found_source_concentration_scale", 1.0)),
        found_source_stay_penalty=float(env_cfg.get("found_source_stay_penalty", 0.0)),
        found_source_stay_radius_scale=float(env_cfg.get("found_source_stay_radius_scale", 1.0)),
        init_pos_mode=str(env_cfg.get("init_pos_mode", "random")),
        seed=seed,
        uav_params=uav_params,
        dynamic_field_config=env_cfg.get("dynamic_field", {}),
    )

    initial_obs = env.reset()
    state_dim = len(initial_obs[0])
    action_dim = int(env_cfg.get("action_dim", 8))
    num_agents = int(env_cfg.get("num_agents", 2))

    agent_cfg = config["agent"]
    agent_algorithm = str(agent_cfg.get("algorithm", "ppo")).lower()
    network_type = str(agent_cfg.get("network", {}).get("type", agent_cfg.get("network_type", "ffnn"))).lower()

    marl_algorithm = str(config["marl"].get("algorithm", "mappo")).lower()
    if marl_algorithm == "mappo" and agent_algorithm != "ppo":
        raise ValueError("MAPPO requires agents to use PPO.")
    if marl_algorithm == "qmix" and agent_algorithm != "dqn":
        raise ValueError("QMIX requires agents to use DQN.")

    agent_hparams = _build_agent_config(agent_cfg)

    agents = [
        MARLAgent(
            agent_id=i,
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=agent_algorithm,
            network_type=network_type,
            config=agent_hparams,
        )
        for i in range(num_agents)
    ]

    if marl_algorithm == "mappo":
        trainer_cfg = config["marl"].get("mappo", {})
    else:
        trainer_cfg = config["marl"].get("qmix", {})
        trainer_cfg = dict(trainer_cfg)
        trainer_cfg["global_state_dim"] = num_agents * state_dim

    trainer = MARLTrainer(agents, env, algorithm=marl_algorithm, config=trainer_cfg)

    output_cfg = config["output"]
    Path(output_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    lr_value = float(agent_hparams.get("lr", 0.0))
    agent_gamma_value = float(agent_hparams.get("gamma", 0.0))
    gamma_value = float(trainer_cfg.get("gamma", agent_gamma_value))
    batch_size = int(agent_hparams.get("batch_size", 0))
    aux_enabled = bool(agent_hparams.get("aux_enabled", False))
    aux_tag = "auxon" if aux_enabled else "auxoff"
    file_tag = (
        f"seed{seed}_agent{agent_algorithm}_net{network_type}_marl{marl_algorithm}_na{num_agents}"
        f"_lr{lr_value:.6f}_gamma{gamma_value:.4f}_bs{batch_size}_{aux_tag}"
    )
    _snapshot_run_artifacts(
        config=config,
        output_cfg=output_cfg,
        file_tag=file_tag,
        train_config_path=args.train_config,
        agent_config_path=args.agent_config,
    )

    log_file = f"{file_tag}.log"
    avg_reward_csv = EXPERIMENT_LOG_DIR / f"{file_tag}_avg_reward_trend.csv"
    logger = _setup_logging(EXPERIMENT_LOG_DIR, log_file)

    pretrained_dir = args.pretrained_dir
    if not pretrained_dir:
        pretrained_dir = str(config.get("training", {}).get("pretrained_dir", "")).strip()

    if pretrained_dir:
        load_path = Path(pretrained_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"Pretrained directory not found: {load_path}")
        trainer.load_models(str(load_path))
        logger.info(f"Loaded pretrained models from {load_path}")

    train_loop(
        trainer,
        env,
        config,
        logger,
        avg_reward_csv,
        checkpoint_run_tag=file_tag,
        log_run_tag=file_tag,
    )


if __name__ == "__main__":
    main()
