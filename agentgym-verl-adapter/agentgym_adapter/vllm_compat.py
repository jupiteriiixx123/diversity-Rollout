"""Minimal vLLM V1 server adapter for official verl v0.5.0.

Scope: the *LLM* server API, not the AgentGym environment server.
- Honor the per-turn ``max_tokens`` passed by BaseAgentGymLoop.
- Cap it by the remaining model context length.
- Request native vLLM LoRA support when the verl actor uses LoRA.

This module does not implement FSDP -> vLLM LoRA weight transfer. That
remains the responsibility of the official verl sharding manager, and its
v0.5.0 V1 compatibility must be verified during the first training run.
"""

from __future__ import annotations

from typing import Any

import ray
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer


def make_sampling_params(sampling_params: dict[str, Any], *, remaining_context: int) -> SamplingParams:
    """Merge caller generation limits without duplicating ``max_tokens``.

    The upstream v0.5.0 server does ``SamplingParams(max_tokens=remaining,
    **sampling_params)``. AgentGym's loop already supplies ``max_tokens``;
    hence upstream raises ``TypeError: multiple values for keyword argument``.
    """
    if remaining_context <= 0:
        raise ValueError("Cannot generate: no tokens left in the model context")
    params = dict(sampling_params)
    requested = int(params.pop("max_tokens", remaining_context))
    if requested <= 0:
        raise ValueError(f"max_tokens must be positive; got {requested}")
    return SamplingParams(max_tokens=min(requested, remaining_context), **params)


# In verl v0.5.0 AsyncvLLMServer is already decorated with @ray.remote.
# Ray ActorClass objects cannot be subclassed directly. Extract its underlying
# Python class and create a new Ray actor using the same public interface.
_UpstreamServerClass = AsyncvLLMServer.__ray_metadata__.modified_class


@ray.remote(num_cpus=1)
class AgentGymAsyncvLLMServer(_UpstreamServerClass):
    """verl's async vLLM server with a token-budget fix and LoRA engine flag."""

    def _create_engine_config(self, engine_args):
        # v0.5.0's AsyncvLLMServer does not forward LoRA configuration to
        # AsyncEngineArgs, even when the trainer creates a PEFT actor.
        lora_rank = int(self.config.model.get("lora_rank", 0))
        if lora_rank > 0:
            engine_args.enable_lora = True
            engine_args.max_lora_rank = lora_rank
            engine_args.max_loras = 1
            engine_args.max_cpu_loras = 1
        return super()._create_engine_config(engine_args)

    async def generate(
        self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str
    ) -> list[int]:
        params = make_sampling_params(
            sampling_params, remaining_context=self.max_model_len - len(prompt_ids)
        )
        generator = self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=params,
            request_id=request_id,
        )
        final_res: RequestOutput | None = None
        async for output in generator:
            final_res = output
        if final_res is None or not final_res.outputs:
            raise RuntimeError(f"vLLM returned no output for request {request_id}")
        return list(final_res.outputs[0].token_ids)
