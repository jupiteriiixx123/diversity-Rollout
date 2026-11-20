"""Real BabyAI server + fake LLM. One environment step; no GPU inference."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agentgym_adapter.babyai_loop import BabyAIAgentLoop
from agentgym_adapter.reward import compute_score

ROOT = Path("/public/home/hh_hh/AgentGym-RL")
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
DATA = ROOT / "agentgym-verl-adapter/data/babyai_train.parquet"


async def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    row = load_dataset("parquet", data_files=str(DATA))["train"][0]

    class FixedActionServer:
        async def generate(self, request_id, *, prompt_ids, sampling_params):
            return tokenizer.encode(
                "Thought:\nI will explore.\n\nAction:\nturn left",
                add_special_tokens=False,
            )

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 2048}}})
    loop = BabyAIAgentLoop(
        trainer_config=SimpleNamespace(config=config),
        server_manager=FixedActionServer(),
        tokenizer=tokenizer,
        client_args={"env_server_base": "http://admin:36001", "data_len": 810},
        max_rounds=1,
        max_tokens_per_turn=128,
    )
    output = await loop.run(row["prompt"], {"temperature": 0.7, "top_p": 1.0})
    assert len(output.response_ids) == len(output.response_mask)
    assert 0 < sum(output.response_mask) < len(output.response_mask)
    assert output.num_turns == 5
    all_ids = output.prompt_ids + output.response_ids
    chatml = tokenizer.decode(all_ids, skip_special_tokens=False)
    assert chatml.count("<|im_start|>system") == 1, "Duplicate system prompt found"
    assert chatml.count("<|im_start|>user") == 3, "Unexpected number of user turns"
    assert chatml.count("<|im_start|>assistant") == 2, "Unexpected assistant turns"
    trajectory = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    score = compute_score("babyai", trajectory)
    print("Task:", row["extra_info"]["item_id"])
    print("Model tokens:", sum(output.response_mask))
    print("Environment tokens:", len(output.response_mask) - sum(output.response_mask))
    print("Environment steps: 1")
    print("Final score:", score)
    print("System prompts: 1")
    print("Real BabyAI loop: OK")


asyncio.run(main())
