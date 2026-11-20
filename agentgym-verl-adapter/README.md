# AgentGym → official verl 0.5.0 adapter (v0.2)

## Design

- `agentgym_adapter/base_loop.py`: shared environment lifecycle, async inference,
  observation handling, Qwen2.5 ChatML token framing, response masks, output.
  **No environment-specific `if` branches.**
- `agentgym_adapter/babyai_loop.py`: thin specialization using AgentGym's
  `env_client_cls` and `env_name` convention; parses BabyAI task index and
  serializes BabyAI's terminal reward.
- `agentgym_adapter/reward.py`: retrieves the per-trajectory terminal score.
- `scripts/prepare_babyai.py`: one-time JSON→Parquet conversion (unchanged).
- `configs/babyai_agent.yaml`: same `_target_` as v0.1.

Both upstream AgentGym and upstream verl are left untouched.

## Update a v0.1 installation

Unzip with `unzip -o`, preserving any local files outside this archive. From
`/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter`, run:

```bash
conda activate verl
unset PYTHONPATH
export TOKENIZERS_PARALLELISM=false
python -m pip install --no-deps -e .
python tests/test_reward.py
python tests/test_chatml.py
python tests/test_loop_live.py
```

The live test uses one *real* BabyAI environment episode with a fixed LLM
response. It needs the BabyAI server at `http://admin:36001` and local Qwen2.5
model tokenizer. It does not start GRPO.

## Training integration (not yet end-to-end verified)

```text
data.train_files=/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter/data/babyai_train.parquet
data.return_raw_chat=True
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.agent.agent_loop_config_path=/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter/configs/babyai_agent.yaml
actor_rollout_ref.rollout.response_length=8192
reward_model.reward_manager=naive
custom_reward_function.path=/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter/agentgym_adapter/reward.py
custom_reward_function.name=compute_score
```

**Important:** `BaseAgentGymLoop` is deliberately Qwen2.5-specific in its
*token framing* at this stage. Changing model families requires checking their
chat templates. It is environment-agnostic, not model-template-agnostic.

**Important:** The final environment score marker is stored in the masked
terminal user message after the last model action; do not use this with
truncation that can remove the marker. A missing marker raises rather than
producing a fabricated reward. Response length overflow fails explicitly.
