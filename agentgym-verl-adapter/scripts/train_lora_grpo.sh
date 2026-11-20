#!/usr/bin/env bash
set -Eeuo pipefail

# 打印日志代码修改位于verl-0.5.0/verl/trainer/ppo/ray_trainer.py 的fit() 中， batch = compute_advantage(...)
# One official verl v0.5.0 GRPO step (4 prompts x 2 trajectories by default).
# Does not modify AgentGym, BaseAgentGymLoop, or the official verl source.
export VERL_LORA_MISSING_POLICY=error
export VERL_VLLM_DISTRIBUTED_BACKEND=zeromq
ROOT="${AGENTGYM_ROOT:-/public/home/hh_hh/AgentGym-RL}"
PROJECT="$ROOT/agentgym-verl-adapter"
MODEL="$ROOT/models/Qwen2.5-7B-Instruct"
DATA="$PROJECT/data/babyai_train.parquet"
AGENT_CONFIG="${AGENT_CONFIG:-$PROJECT/configs/babyai_one_step_agent.yaml}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/train_810_lora}"

GPU_IDS="${GPU_IDS:-0,2,3,4}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
N_GPUS="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# These flags must be visible to both the driver and its Ray workers.
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/${USER}/agentgym_hf_datasets}"
export RAY_prestart_worker_first_driver=0
mkdir -p "$HF_DATASETS_CACHE" "$RUN_DIR"

[[ -f "$DATA" ]] || { echo "Missing training Parquet: $DATA" >&2; exit 1; }
[[ -f "$AGENT_CONFIG" ]] || { echo "Missing agent config: $AGENT_CONFIG" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "Missing model directory: $MODEL" >&2; exit 1; }
[[ "$N_GPUS" -gt 0 && "$BATCH_SIZE" -gt 0 && "$ROLLOUT_N" -gt 1 ]] || {
  echo "Expected GPU_IDS, BATCH_SIZE > 0, ROLLOUT_N > 1" >&2; exit 1;
}

# The user runs this within the tested (verl) conda environment.
# In particular, DO NOT prepend the legacy AgentGym-RL verl fork to PYTHONPATH.
unset PYTHONPATH
cd "$PROJECT"

echo "[AgentGym/verl] GPU_IDS=$GPU_IDS; prompts=$BATCH_SIZE; n=$ROLLOUT_N; LoRA rank=$LORA_RANK"
echo "[AgentGym/verl] Run directory: $RUN_DIR"

RAY_ADDRESS=local python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers="$BATCH_SIZE" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CONFIG" \
    actor_rollout_ref.rollout.agent.custom_async_server.path=pkg://agentgym_adapter.vllm_compat \
    actor_rollout_ref.rollout.agent.custom_async_server.name=AgentGymAsyncvLLMServer \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path="$PROJECT/agentgym_adapter/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=120 \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RUN_DIR/checkpoints/global_step_30" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.rollout_data_dir=/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter/runs/rl_audit/rollouts \
    trainer.logger='[console]' \
    trainer.project_name=agentgym_verl \
    trainer.experiment_name=babyai_grpo_120 \
    ray_init.num_cpus=16 \
    2>&1 | tee "$RUN_DIR/train.log"
