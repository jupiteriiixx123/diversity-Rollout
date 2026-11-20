"""Extract the terminal BabyAI environment score from the final (masked) observation.

This adapter targets official verl v0.5.0, whose AgentLoopOutput does not
contain a reward_score field.  The marker is appended only after the last LLM
generation and is never passed to a later model call.
"""

import re

SCORE_PREFIX = "[[BABYAI_FINAL_SCORE="
SCORE_SUFFIX = "]]"
_SCORE_RE = re.compile(r"\[\[BABYAI_FINAL_SCORE=(-?(?:\d+(?:\.\d*)?|\.\d+))\]\]")


def format_final_score(score: float) -> str:
    return f"{SCORE_PREFIX}{float(score):.8f}{SCORE_SUFFIX}"


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    """Compatible with verl's reward_model.custom_reward_function hook."""
    if data_source != "babyai":
        raise ValueError(f"Expected data_source=babyai, got {data_source!r}")
    matches = _SCORE_RE.findall(solution_str)
    if not matches:
        raise ValueError("Missing final BabyAI score marker in rollout. Refusing to train with a fabricated reward.")
    return float(matches[-1])
