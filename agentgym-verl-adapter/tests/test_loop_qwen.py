"""Real Qwen2.5-7B + real BabyAI multi-turn test against the frozen v0.2.1 loop.

Run: CUDA_VISIBLE_DEVICES=0 python tests/test_loop_qwen.py
This uses a standalone vLLM V0 engine (NOT verl's distributed async server).
It tests real generation and environment interaction, not FSDP/LoRA sync.
"""

import os

# Must be set before importing vllm/tokenizers.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agentgym_adapter.babyai_loop import BabyAIAgentLoop
from agentgym_adapter.reward import compute_score

ROOT = Path("/public/home/hh_hh/AgentGym-RL")
PROJECT = ROOT / "agentgym-verl-adapter"
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
DATA = PROJECT / "data/babyai_train.parquet"
TASK_ROW = int(os.getenv("BABYAI_TASK_ROW", "0"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
MAX_TOKENS_PER_TURN = int(os.getenv("MAX_TOKENS_PER_TURN", "128"))
RESPONSE_LENGTH = int(os.getenv("RESPONSE_LENGTH", "4096"))


async def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    dataset = load_dataset("parquet", data_files=str(DATA))["train"]
    row = dataset[TASK_ROW]
    task_id = row["extra_info"]["item_id"]

    # Real vLLM inference with the existing local 7B model.
    llm = LLM(
        model=str(MODEL),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.55")),
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    class VLLMServerAdapter:
        """Tiny async facade around a real standalone vLLM LLM instance."""

        def __init__(self):
            self.actions = []

        async def generate(self, request_id, *, prompt_ids, sampling_params):
            params = SamplingParams(
                temperature=float(sampling_params.get("temperature", 0.7)),
                top_p=float(sampling_params.get("top_p", 0.95)),
                repetition_penalty=float(sampling_params.get("repetition_penalty", 1.0)),
                max_tokens=int(sampling_params["max_tokens"]),
                stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
            )

            def infer():
                results = llm.generate(
                    prompt_token_ids=[prompt_ids],
                    sampling_params=params,
                    use_tqdm=False,
                )
                return list(results[0].outputs[0].token_ids)

            tokens = await asyncio.to_thread(infer)
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            self.actions.append({"input_tokens": len(prompt_ids), "output_tokens": len(tokens), "text": answer})
            print(f"\n=== Qwen action {len(self.actions)} ({len(tokens)} tokens) ===\n{answer[:1200]}", flush=True)
            return tokens

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": RESPONSE_LENGTH}}})
    server = VLLMServerAdapter()
    agent = BabyAIAgentLoop(
        trainer_config=SimpleNamespace(config=config),
        server_manager=server,
        tokenizer=tokenizer,
        client_args={"env_server_base": os.getenv("BABYAI_URL", "http://admin:36001"), "data_len": len(dataset)},
        max_rounds=MAX_ROUNDS,
        max_tokens_per_turn=MAX_TOKENS_PER_TURN,
    )

    output = await agent.run(row["prompt"], {"temperature": 0.7, "top_p": 0.95})
    assert len(output.response_ids) == len(output.response_mask)
    assert 0 < sum(output.response_mask) < len(output.response_mask)
    assert len(server.actions) >= 1
    assert output.num_turns == 3 + 2 * len(server.actions)

    full_chatml = tokenizer.decode(output.prompt_ids + output.response_ids, skip_special_tokens=False)
    assert full_chatml.count("<|im_start|>system") == 1, "Repeated system prompt"
    decoded_response = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    score = compute_score("babyai", decoded_response)

    result = {
        "task": task_id,
        "model_actions": len(server.actions),
        "model_tokens": sum(output.response_mask),
        "environment_tokens": len(output.response_mask) - sum(output.response_mask),
        "response_tokens": len(output.response_ids),
        "final_score": score,
        "actions": server.actions,
    }
    logs = PROJECT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out_path = logs / f"real_qwen_{task_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Final ===")
    print(json.dumps({k: v for k, v in result.items() if k != "actions"}, indent=2))
    print("Trace saved:", out_path)
    print("Real Qwen + BabyAI loop: OK")


if __name__ == "__main__":
    asyncio.run(main())