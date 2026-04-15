"""Dynamic episode visualisation utilities for CCUS-Gym."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np

from ccus_gym.rl.mappo import RoleMAPPOPolicy
from ccus_gym.rl.training import build_role_groups
from ccus_gym.sim.env import CCUSEnv


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return { _to_builtin(k): _to_builtin(v) for k, v in value.items() }
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _role_from_agent(agent: str) -> str:
    if agent.startswith("emitter_"):
        return "emitter"
    if agent.startswith("transport_"):
        return "transport"
    if agent.startswith("storage_"):
        return "storage"
    raise ValueError(f"Unrecognized agent name: {agent}")


def rollout_episode_trace(
    env: CCUSEnv,
    policies: Optional[Dict[str, RoleMAPPOPolicy]] = None,
    *,
    seed: int = 42,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one episode and record per-step decision and outcome traces."""
    observations, _ = env.reset(seed=seed)
    frames: List[Dict[str, Any]] = []
    done = False

    while not done:
        state = env.global_state_vector()
        actions: Dict[str, np.ndarray] = {}
        for agent in env.agents:
            if policies is None:
                actions[agent] = env.action_space(agent).sample()
            else:
                role = _role_from_agent(agent)
                action, _, _ = policies[role].act(
                    observations[agent],
                    state,
                    deterministic=deterministic,
                )
                actions[agent] = action

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        frame_info = dict(infos[env.agents[0]]) if env.agents else {}
        frame_info["agent_actions"] = {
            agent: [float(x) for x in np.asarray(action).tolist()]
            for agent, action in actions.items()
        }
        frame_info["agent_rewards"] = {
            agent: float(reward) for agent, reward in rewards.items()
        }
        frames.append(_to_builtin(frame_info))
        observations = next_obs
        if max_steps is not None and len(frames) >= max_steps:
            break

    return {
        "frames": frames,
        "episode_stats": env.get_episode_stats(),
        "config": {
            "episode_length": env.episode_length,
            "transport_modes": list(env._transport_modes),  # type: ignore[attr-defined]
        },
    }


def save_episode_trace_json(path: str, trace: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_to_builtin(trace), indent=2), encoding="utf-8")


def _build_layout(env: CCUSEnv) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    emitter_ids = sorted(env.physical_layer.emitters.keys())
    storage_ids = sorted(env.physical_layer.storage_sites.keys())
    transport_modes = list(env._transport_modes)  # type: ignore[attr-defined]

    emitter_ys = np.linspace(0.15, 0.85, max(len(emitter_ids), 1))
    transport_ys = np.linspace(0.2, 0.8, max(len(transport_modes), 1))
    storage_ys = np.linspace(0.15, 0.85, max(len(storage_ids), 1))

    for idx, eid in enumerate(emitter_ids):
        positions[f"emitter_{eid}"] = (0.12, float(emitter_ys[idx]))
    for idx, mode_name in enumerate(transport_modes):
        positions[f"transport_{mode_name}"] = (0.5, float(transport_ys[idx]))
    for idx, sid in enumerate(storage_ids):
        positions[f"storage_{sid}"] = (0.88, float(storage_ys[idx]))
    return positions


def _route_edge_maps(env: CCUSEnv) -> Tuple[Dict[Tuple[int, str], float], Dict[Tuple[str, int], float]]:
    emitter_to_transport: Dict[Tuple[int, str], float] = {}
    transport_to_storage: Dict[Tuple[str, int], float] = {}
    for eid in sorted(env.physical_layer.emitters.keys()):
        for transport_idx, sid in env.physical_layer.get_routes_for_emitter(eid):
            mode_name = env._transport_modes[transport_idx]  # type: ignore[attr-defined]
            emitter_to_transport[(eid, mode_name)] = 0.0
            transport_to_storage[(mode_name, sid)] = 0.0
    return emitter_to_transport, transport_to_storage


def _prepare_frame_series(
    env: CCUSEnv,
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    frames = trace["frames"]
    layout = _build_layout(env)
    emitter_to_transport_base, transport_to_storage_base = _route_edge_maps(env)
    cumulative_stored: List[float] = []
    cumulative_vented: List[float] = []
    running_stored = 0.0
    running_vented = 0.0

    prepared_frames: List[Dict[str, Any]] = []
    max_route_flow = 1e-6
    max_step_volume = 1e-6
    max_pressure_frac = 1.0

    for frame in frames:
        decisions = frame.get("decision_summary", {})
        outcome = frame.get("outcome_summary", {})
        physical_state = frame.get("physical_state", {})

        emitter_to_transport = dict(emitter_to_transport_base)
        transport_to_storage = dict(transport_to_storage_base)

        for eid, emitter_data in decisions.get("emitters", {}).items():
            for route in emitter_data.get("routes", []):
                key_left = (int(eid), str(route["transport_mode"]))
                key_right = (str(route["transport_mode"]), int(route["storage_id"]))
                volume = float(route["volume"])
                emitter_to_transport[key_left] = emitter_to_transport.get(key_left, 0.0) + volume
                transport_to_storage[key_right] = transport_to_storage.get(key_right, 0.0) + volume
                max_route_flow = max(max_route_flow, volume)

        if emitter_to_transport:
            max_route_flow = max(max_route_flow, max(emitter_to_transport.values()))
        if transport_to_storage:
            max_route_flow = max(max_route_flow, max(transport_to_storage.values()))

        step_stored = float(frame.get("step_stored", 0.0))
        step_vented = float(frame.get("step_vented", 0.0))
        step_captured = float(frame.get("step_captured", 0.0))
        running_stored += step_stored
        running_vented += step_vented
        cumulative_stored.append(running_stored)
        cumulative_vented.append(running_vented)
        max_step_volume = max(max_step_volume, step_stored, step_vented, step_captured)

        storage_states = physical_state.get("storage_sites", {})
        for sdata in storage_states.values():
            max_pressure_frac = max(max_pressure_frac, float(sdata.get("pressure_frac", 0.0)))

        prepared_frames.append(
            {
                "frame": frame,
                "decisions": decisions,
                "outcome": outcome,
                "physical_state": physical_state,
                "emitter_to_transport": emitter_to_transport,
                "transport_to_storage": transport_to_storage,
                "step_stored": step_stored,
                "step_vented": step_vented,
                "step_captured": step_captured,
            }
        )

    return {
        "layout": layout,
        "frames": prepared_frames,
        "cumulative_stored": cumulative_stored,
        "cumulative_vented": cumulative_vented,
        "max_route_flow": max_route_flow,
        "max_step_volume": max_step_volume,
        "max_pressure_frac": max_pressure_frac,
    }


def animate_episode_trace(
    env: CCUSEnv,
    trace: Dict[str, Any],
    *,
    title: str = "CCUS Decision Replay",
    interval_ms: int = 1200,
) -> animation.FuncAnimation:
    prepared = _prepare_frame_series(env, trace)
    frames = prepared["frames"]
    layout = prepared["layout"]
    cumulative_stored = prepared["cumulative_stored"]
    cumulative_vented = prepared["cumulative_vented"]
    max_route_flow = prepared["max_route_flow"]
    max_step_volume = prepared["max_step_volume"]
    max_pressure_frac = prepared["max_pressure_frac"]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_network = fig.add_subplot(gs[:, 0])
    ax_cumulative = fig.add_subplot(gs[0, 1])
    ax_step = fig.add_subplot(gs[1, 1])

    cmap = get_cmap("YlOrRd")
    pressure_norm = Normalize(vmin=0.0, vmax=max(1.05, max_pressure_frac))

    emitter_ids = sorted(env.physical_layer.emitters.keys())
    storage_ids = sorted(env.physical_layer.storage_sites.keys())
    transport_modes = list(env._transport_modes)  # type: ignore[attr-defined]

    def _draw_static_network() -> None:
        ax_network.clear()
        ax_network.set_xlim(0.0, 1.0)
        ax_network.set_ylim(0.0, 1.0)
        ax_network.axis("off")
        ax_network.set_title(f"{title}\nDecision Network")

    def _update(frame_idx: int) -> List[Any]:
        prepared_frame = frames[frame_idx]
        frame = prepared_frame["frame"]
        decisions = prepared_frame["decisions"]
        outcome = prepared_frame["outcome"]
        physical_state = prepared_frame["physical_state"]

        _draw_static_network()
        artists: List[Any] = []

        for (eid, mode_name), volume in prepared_frame["emitter_to_transport"].items():
            start = layout[f"emitter_{eid}"]
            end = layout[f"transport_{mode_name}"]
            flow_ratio = min(1.0, volume / max_route_flow)
            width = 1.0 + 10.0 * flow_ratio
            alpha = 0.15 + 0.75 * flow_ratio
            line = ax_network.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#7aa6c2",
                linewidth=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=1,
            )[0]
            artists.append(line)

        for (mode_name, sid), volume in prepared_frame["transport_to_storage"].items():
            start = layout[f"transport_{mode_name}"]
            end = layout[f"storage_{sid}"]
            flow_ratio = min(1.0, volume / max_route_flow)
            width = 1.0 + 10.0 * flow_ratio
            alpha = 0.15 + 0.75 * flow_ratio
            line = ax_network.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#68b684",
                linewidth=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=1,
            )[0]
            artists.append(line)

        for eid in emitter_ids:
            pos = layout[f"emitter_{eid}"]
            est = physical_state.get("emitters", {}).get(eid, {})
            capture_frac = decisions.get("emitters", {}).get(eid, {}).get("capture_frac", 0.0)
            buffer_frac = float(est.get("buffer_frac", 0.0))
            color = (0.15, 0.45 + 0.45 * capture_frac, 0.85 - 0.55 * buffer_frac)
            marker = ax_network.scatter(
                [pos[0]],
                [pos[1]],
                s=1100,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
            )
            label = ax_network.text(
                pos[0],
                pos[1],
                f"E{eid}\ncap {capture_frac:.2f}\nbuf {buffer_frac:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )
            artists.extend([marker, label])

        for mode_name in transport_modes:
            pos = layout[f"transport_{mode_name}"]
            tstate = physical_state.get("transports", {}).get(mode_name, {})
            utilization = float(tstate.get("utilization", 0.0))
            quality_threshold = decisions.get("transport", {}).get(mode_name, {}).get(
                "quality_threshold",
                0.0,
            )
            marker = ax_network.scatter(
                [pos[0]],
                [pos[1]],
                s=1400,
                color=(0.95, 0.85 - 0.5 * utilization, 0.25 + 0.5 * utilization),
                marker="s",
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
            )
            label = ax_network.text(
                pos[0],
                pos[1],
                f"{mode_name}\nutil {utilization:.2f}\nq>{quality_threshold:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )
            artists.extend([marker, label])

        for sid in storage_ids:
            pos = layout[f"storage_{sid}"]
            sstate = physical_state.get("storage_sites", {}).get(sid, {})
            pressure_frac = float(sstate.get("pressure_frac", 0.0))
            inlet_purity = outcome.get("storage", {}).get(sid, {}).get("inlet_purity", 0.0)
            quality_flag = outcome.get("storage", {}).get(sid, {}).get("quality_violation", False)
            marker = ax_network.scatter(
                [pos[0]],
                [pos[1]],
                s=1300,
                color=cmap(pressure_norm(pressure_frac)),
                marker="o",
                edgecolor="crimson" if quality_flag else "black",
                linewidth=2.0 if quality_flag else 1.2,
                zorder=3,
            )
            label = ax_network.text(
                pos[0],
                pos[1],
                f"S{sid}\np {pressure_frac:.2f}\nco2 {inlet_purity:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )
            artists.extend([marker, label])

        disruptions = frame.get("disruptions_detail", [])
        disruption_lines = ["No disruptions"] if not disruptions else [
            f"{d['cause']} -> {d['target_type']} {d['target_id']} ({d['severity']:.2f})"
            for d in disruptions[:4]
        ]
        metrics_text = (
            f"Month {frame.get('timestep', frame_idx + 1)}\n"
            f"Step captured: {prepared_frame['step_captured']:.3f} Mt\n"
            f"Step stored: {prepared_frame['step_stored']:.3f} Mt\n"
            f"Step vented: {prepared_frame['step_vented']:.3f} Mt\n"
            f"Step quality violations: {frame.get('step_quality_violations', 0)}\n"
            f"Carbon tax: {frame.get('economic_context', {}).get('carbon_tax', 0.0):.1f}\n"
            f"Electricity: {frame.get('economic_context', {}).get('electricity_price', 0.0):.1f}\n"
            f"Disruptions:\n- " + "\n- ".join(disruption_lines)
        )
        text_artist = ax_network.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax_network.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
            zorder=5,
        )
        artists.append(text_artist)

        ax_cumulative.clear()
        ax_cumulative.set_title("Cumulative System Response")
        x = np.arange(1, len(frames) + 1)
        ax_cumulative.plot(x, cumulative_stored, color="#2a9d8f", linewidth=2.5, label="stored")
        ax_cumulative.plot(x, cumulative_vented, color="#e76f51", linewidth=2.5, label="vented")
        ax_cumulative.scatter(
            [frame_idx + 1],
            [cumulative_stored[frame_idx]],
            color="#2a9d8f",
            s=70,
            zorder=5,
        )
        ax_cumulative.scatter(
            [frame_idx + 1],
            [cumulative_vented[frame_idx]],
            color="#e76f51",
            s=70,
            zorder=5,
        )
        ax_cumulative.set_xlim(1, max(2, len(frames)))
        ymax = max(max(cumulative_stored, default=0.0), max(cumulative_vented, default=0.0), 0.1)
        ax_cumulative.set_ylim(0.0, ymax * 1.15)
        ax_cumulative.set_xlabel("Month")
        ax_cumulative.set_ylabel("MtCO2")
        ax_cumulative.grid(alpha=0.25)
        ax_cumulative.legend(loc="upper left")

        ax_step.clear()
        ax_step.set_title("Current Step Decisions and Constraints")
        step_labels = ["captured", "stored", "vented"]
        step_values = [
            prepared_frame["step_captured"],
            prepared_frame["step_stored"],
            prepared_frame["step_vented"],
        ]
        bar_positions = np.arange(len(step_labels))
        bars = ax_step.bar(
            bar_positions,
            step_values,
            color=["#457b9d", "#2a9d8f", "#e76f51"],
            width=0.55,
        )
        artists.extend(list(bars))

        pressure_x = np.arange(len(storage_ids)) + len(step_labels) + 1
        pressure_vals = [
            float(physical_state.get("storage_sites", {}).get(sid, {}).get("pressure_frac", 0.0))
            for sid in storage_ids
        ]
        pressure_colors = [cmap(pressure_norm(v)) for v in pressure_vals]
        p_bars = ax_step.bar(
            pressure_x,
            pressure_vals,
            color=pressure_colors,
            width=0.55,
        )
        artists.extend(list(p_bars))

        quality_x = np.arange(len(transport_modes)) + len(step_labels) + len(storage_ids) + 2
        quality_vals = [
            float(decisions.get("transport", {}).get(mode_name, {}).get("quality_threshold", 0.0))
            for mode_name in transport_modes
        ]
        q_bars = ax_step.bar(
            quality_x,
            quality_vals,
            color="#8d99ae",
            width=0.55,
        )
        artists.extend(list(q_bars))

        tick_positions = list(bar_positions) + list(pressure_x) + list(quality_x)
        tick_labels = step_labels + [f"S{sid} p" for sid in storage_ids] + [f"{mode} q" for mode in transport_modes]
        ax_step.set_xticks(tick_positions)
        ax_step.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax_step.set_ylim(0.0, max(max_step_volume * 1.25, max_pressure_frac * 1.25, 1.0))
        ax_step.grid(axis="y", alpha=0.25)
        ax_step.set_ylabel("Value")

        return artists

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return ani


def save_episode_animation(
    path: str,
    env: CCUSEnv,
    trace: Dict[str, Any],
    *,
    title: str = "CCUS Decision Replay",
    interval_ms: int = 1200,
) -> None:
    """Save the decision replay animation to an HTML file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani = animate_episode_trace(
        env,
        trace,
        title=title,
        interval_ms=interval_ms,
    )
    html = (
        "<html><head><meta charset='utf-8'><title>CCUS Decision Replay</title></head>"
        "<body style='font-family: sans-serif; background: #f8fafc; margin: 0; padding: 24px;'>"
        f"<h2 style='margin-top: 0;'>{title}</h2>"
        f"{ani.to_jshtml()}"
        "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")
    plt.close(ani._fig)
