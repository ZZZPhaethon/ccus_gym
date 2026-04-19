"""Local Qwen3 policy: loads model directly via HuggingFace transformers.

No separate HTTP service needed. Designed to run on a GPU node via SLURM.

Supports any causal-LM model on HuggingFace (Qwen3-7B, Qwen3-4B, etc.).
Has the same interface as LLMEmitterPolicy so hybrid_runner works unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ccus_gym.llm.emitter_policy import (
    _SYSTEM_PROMPT,
    _build_user_message,
    _parse_response,
)


class LocalLLMEmitterPolicy:
    """Emitter policy backed by a locally-loaded HuggingFace causal LM.

    The model is loaded once (shared across all emitter agents via a
    class-level registry) to avoid loading multiple copies on the GPU.

    Args:
        agent_id: Agent name, e.g. "emitter_0".
        n_routes: Number of transport routes available to this emitter.
        action_dim: Size of the action vector.
        model_name: HuggingFace model ID or local path.
        call_interval: Steps between LLM calls (actions cached in between).
        max_new_tokens: Max tokens to generate per call.
        temperature: Sampling temperature (0 = greedy).
        device_map: Passed to from_pretrained; "auto" distributes across GPUs.
        load_in_4bit: Enable 4-bit quantization (requires bitsandbytes).
        load_in_8bit: Enable 8-bit quantization (requires bitsandbytes).
    """

    # Shared model/tokenizer registry keyed by model_name to avoid reloading
    _model_registry: Dict[str, Any] = {}
    _tokenizer_registry: Dict[str, Any] = {}

    def __init__(
        self,
        agent_id: str,
        n_routes: int,
        action_dim: int,
        *,
        model_name: str = "Qwen/Qwen3-7B",
        call_interval: int = 12,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.n_routes = n_routes
        self.action_dim = action_dim
        self.model_name = model_name
        self.call_interval = call_interval
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        self._cached_action: Optional[np.ndarray] = None
        self._step_count: int = 0
        self.call_log: List[Dict[str, Any]] = []

        # Lazy-load model on first use
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Load model and tokenizer (once per model_name across all agents)."""
        if self.model_name in self.__class__._model_registry:
            self._model = self.__class__._model_registry[self.model_name]
            self._tokenizer = self.__class__._tokenizer_registry[self.model_name]
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for local LLM inference. "
                "Install with: pip install transformers accelerate"
            ) from e

        print(f"[LocalLLMEmitterPolicy] Loading {self.model_name} …")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                )
                load_kwargs["quantization_config"] = bnb_cfg
            except ImportError:
                print(
                    "[LocalLLMEmitterPolicy] bitsandbytes not found, "
                    "loading in bf16 instead of quantized."
                )
                load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        model.eval()

        self.__class__._model_registry[self.model_name] = model
        self.__class__._tokenizer_registry[self.model_name] = tokenizer
        self._model = model
        self._tokenizer = tokenizer
        print(f"[LocalLLMEmitterPolicy] {self.model_name} loaded.")

    def reset(self) -> None:
        self._cached_action = None
        self._step_count = 0

    def act(self, env_context: Dict[str, Any]) -> np.ndarray:
        """Return action, calling LLM every call_interval steps."""
        if self._cached_action is None or self._step_count % self.call_interval == 0:
            if self._model is None:
                self._load_model()
            action, reasoning = self._query_local(env_context)
            self._cached_action = action
            self.call_log.append({
                "timestep": env_context.get("timestep"),
                "reasoning": reasoning,
                "action": action.tolist(),
            })
        self._step_count += 1
        return self._cached_action.copy()

    def _query_local(self, env_context: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Run one inference pass with the local model."""
        import torch

        user_msg = _build_user_message(env_context)

        # Build chat template (Qwen3 uses standard chat format)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Qwen3 supports enable_thinking=False for faster non-thinking responses
        tokenize_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer version without enable_thinking
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, **tokenize_kwargs)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        try:
            action, reasoning = _parse_response(response_text, self.n_routes, self.action_dim)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            action = np.full(self.action_dim, 0.5, dtype=np.float32)
            reasoning = f"[parse error: {e}] raw: {response_text[:100]}"

        return action, reasoning
