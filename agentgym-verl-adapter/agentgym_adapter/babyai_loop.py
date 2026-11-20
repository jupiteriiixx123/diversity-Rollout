"""BabyAI-specific hooks for the shared AgentGym/verl AgentLoop."""

import re

from agentenv.envs.babyai import BabyAIEnvClient

from .base_loop import BaseAgentGymLoop
from .reward import format_final_score


class BabyAIAgentLoop(BaseAgentGymLoop):
    # Same subclass convention as AgentGym's BabyAITask(BaseTask).
    env_client_cls = BabyAIEnvClient
    env_name = "BabyAI"

    def parse_task_id(self, messages: list[dict]) -> int:
        item_id = messages[0].get("item_id") if messages else None
        match = re.fullmatch(r"babyai_(\d+)", item_id or "")
        if not match:
            raise ValueError(f"Invalid BabyAI item_id: {item_id!r}")
        return int(match.group(1))

    def terminal_feedback(self, step_result) -> str:
        return "\n" + format_final_score(step_result.reward)
