"""LLM-based emitter policy using any OpenAI-compatible API.

Works with:
  - Qwen3 via DashScope:  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
  - Qwen3 via vLLM:       base_url="http://localhost:8000/v1"
  - Qwen3 via Ollama:     base_url="http://localhost:11434/v1"
  - Any other model that exposes OpenAI-compatible /v1/chat/completions
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_SYSTEM_PROMPT = """You are an expert operations manager for a CO2 Capture, Utilization and Storage (CCUS) network.
Your job is to make strategic monthly decisions for a CO2 emitter facility to maximise carbon storage while managing costs and quality constraints.

Key facts:
- If the buffer overflows, CO2 is vented to atmosphere (very bad — heavy penalty).
- CO2 must reach ≥93% purity for storage sites to accept it. Below threshold = quality violation.
- Higher purification_effort improves purity but increases electricity cost.
- Pipeline transport is cheap (~$8-10/t) but limited capacity.
- Ship/rail transport is more expensive (~$18-25/t) but higher capacity.
- Storage site pressure rises with cumulative injection; when it approaches the limit, injection stops.
- The planning horizon is 120 months (10 years). Balance short-term costs with long-term storage.

You must respond with ONLY a valid JSON object (no markdown, no extra text) in exactly this format:
{
  "reasoning": "<1-2 sentence explanation of your decision>",
  "route_preferences": [<float 0-1 per route, higher = prefer this route>],
  "send_fraction": <float 0-1, fraction of available buffer to dispatch this month>,
  "capture_fraction": <float 0-1, maps to actual capture 10%-100%>,
  "purification_effort": <float 0-1, 0=no extra purification, 1=maximum>
}"""


def _build_user_message(ctx: Dict[str, Any]) -> str:
    es = ctx["emitter_state"]
    econ = ctx["economic"]
    timestep = ctx["timestep"]
    episode_length = ctx["episode_length"]
    route_modes = ctx["route_modes"]

    buf_pct = es["buffer_frac"] * 100
    cap_pct = es["capture_fraction"] * 100
    purity_pct = es["effective_purity"] * 100
    disrupted = "⚠ DISRUPTED" if es["is_disrupted"] > 0.5 else "operating normally"

    lines = [
        f"=== Emitter Status (Month {timestep + 1}/{episode_length}) ===",
        f"Buffer: {buf_pct:.0f}% full  |  Capture rate: {cap_pct:.0f}%  |  CO2 purity: {purity_pct:.1f}%",
        f"Facility: {disrupted}",
        "",
        "Transport routes available:",
    ]

    for i, (mode, ts) in enumerate(zip(route_modes, ctx["transport_states"])):
        cap_avail = ts.get("available_capacity", 0.0)
        cap_total = ts.get("capacity", 1e-9)
        avail_pct = 100.0 * cap_avail / max(cap_total, 1e-9)
        t_disrupted = "⚠ DISRUPTED" if ts.get("is_disrupted", 0.0) > 0.5 else "OK"
        cost = ctx["transport_costs"].get(mode, "?")
        lines.append(
            f"  Route {i} ({mode}): {avail_pct:.0f}% capacity available, "
            f"status={t_disrupted}, approx cost=${cost}/tCO2"
        )

    lines += ["", "Storage sites:"]
    for sid, ss in ctx["storage_states"].items():
        p_pct = ss.get("pressure_frac", 0.0) * 100
        s_disrupted = "⚠ DISRUPTED" if ss.get("is_disrupted", 0.0) > 0.5 else "OK"
        last_purity = ss.get("last_inlet_purity", 1.0) * 100
        lines.append(
            f"  Site {sid}: pressure={p_pct:.0f}% of limit, "
            f"last accepted purity={last_purity:.1f}%, status={s_disrupted}"
        )

    carbon_tax = econ.get("carbon_tax", 0.0)
    elec_price = econ.get("electricity_price", 0.0)
    lines += [
        "",
        f"Economics: carbon tax=${carbon_tax:.0f}/tCO2, electricity=${elec_price:.0f}/MWh",
        "",
        f"Number of routes: {len(route_modes)}. "
        f"Provide route_preferences as a list of {len(route_modes)} floats.",
    ]

    return "\n".join(lines)


def _parse_response(text: str, n_routes: int, action_dim: int) -> Tuple[np.ndarray, str]:
    """Extract JSON from LLM response, return (action_array, reasoning)."""
    # Strip Qwen3 <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract first JSON object from response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {text[:200]}")

    data = json.loads(match.group())

    reasoning = str(data.get("reasoning", ""))

    route_prefs = data.get("route_preferences", [0.5] * n_routes)
    if not isinstance(route_prefs, list) or len(route_prefs) != n_routes:
        route_prefs = [0.5] * n_routes
    route_prefs = [float(np.clip(v, 0.0, 1.0)) for v in route_prefs]

    send_frac = float(np.clip(data.get("send_fraction", 0.7), 0.0, 1.0))
    cap_frac = float(np.clip(data.get("capture_fraction", 0.8), 0.0, 1.0))
    purif = float(np.clip(data.get("purification_effort", 0.2), 0.0, 1.0))

    action = np.zeros(action_dim, dtype=np.float32)
    action[:n_routes] = route_prefs
    if n_routes < action_dim:
        action[n_routes] = send_frac
    if n_routes + 1 < action_dim:
        action[n_routes + 1] = cap_frac
    if n_routes + 2 < action_dim:
        action[n_routes + 2] = purif

    return action, reasoning


class LLMEmitterPolicy:
    """LLM-based policy for emitter agents using any OpenAI-compatible endpoint.

    Calls the LLM every `call_interval` timesteps and caches the action
    in between, avoiding per-step API latency.
    """

    def __init__(
        self,
        agent_id: str,
        n_routes: int,
        action_dim: int,
        *,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3",
        api_key: str = "none",
        call_interval: int = 12,
        temperature: float = 0.3,
        timeout: int = 60,
    ) -> None:
        self.agent_id = agent_id
        self.n_routes = n_routes
        self.action_dim = action_dim
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.call_interval = call_interval
        self.temperature = temperature
        self.timeout = timeout

        self._cached_action: Optional[np.ndarray] = None
        self._step_count: int = 0
        self.call_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._cached_action = None
        self._step_count = 0

    def act(self, env_context: Dict[str, Any]) -> np.ndarray:
        """Return action for this timestep, calling LLM every call_interval steps."""
        if self._cached_action is None or self._step_count % self.call_interval == 0:
            action, reasoning = self._query_llm(env_context)
            self._cached_action = action
            self.call_log.append({
                "timestep": env_context.get("timestep"),
                "reasoning": reasoning,
                "action": action.tolist(),
            })
        self._step_count += 1
        return self._cached_action.copy()

    def _query_llm(self, env_context: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Call the LLM API and parse the response into an action array."""
        user_msg = _build_user_message(env_context)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM API error {e.code} for agent {self.agent_id}: {err_body[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LLM endpoint unreachable ({self.base_url}): {e.reason}. "
                "Start your LLM service or check --llm-base-url."
            ) from e

        text = result["choices"][0]["message"]["content"]
        try:
            action, reasoning = _parse_response(text, self.n_routes, self.action_dim)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            # Fallback: neutral action if parsing fails
            action = np.full(self.action_dim, 0.5, dtype=np.float32)
            reasoning = f"[parse error: {e}] raw: {text[:100]}"

        return action, reasoning
