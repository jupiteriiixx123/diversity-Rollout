#!/usr/bin/env bash
set -Eeuo pipefail

# 新增pass@k metric，修改参数如下
# data.val_batch_size=1 \
# actor_rollout_ref.rollout.val_kwargs.n=4 \
# actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
# actor_rollout_ref.rollout.val_kwargs.do_sample=True \

# Unified BabyAI evaluation for verl v0.5.0.
# Usage:
#   bash scripts/eval/eval_babyai.sh base
#   CKPT=/public/home/hh_hh/AgentGym-RL/agentgym-verl-adapter/runs/babyai_full_epoch_v2/checkpoints/global_step_202 bash scripts/eval/eval_babyai.sh lora
#   CKPT=runs/babyai_epoch3_v3/checkpoints/global_step_606 bash scripts/eval/eval_babyai.sh both

MODE="${1:-both}"                         # base | lora | both
ROOT="${AGENTGYM_ROOT:-/public/home/hh_hh/AgentGym-RL}"
PROJECT="$ROOT/agentgym-verl-adapter"
MODEL="$ROOT/models/Qwen2.5-7B-Instruct"
TRAIN_FILE="$PROJECT/data/babyai_train.parquet"
VAL_FILE="${VAL_FILE:-$PROJECT/data/babyai_test.parquet}"
AGENT_CONFIG="${AGENT_CONFIG:-$PROJECT/configs/babyai_one_step_agent.yaml}"
CKPT="${CKPT:-}"                          # Full path to global_step_N for LoRA
EVAL_NAME="${EVAL_NAME:-babyai_test_v3}"
OUT_ROOT="$PROJECT/runs/eval_babyai/$EVAL_NAME"
GPU_IDS="${GPU_IDS:-0,2,3,4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

[[ "$MODE" == base || "$MODE" == lora || "$MODE" == both ]] || {
  echo "Usage: $0 {base|lora|both}" >&2; exit 2;
}
for path in "$MODEL" "$AGENT_CONFIG" "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
N_GPUS="${#GPU_ARRAY[@]}"
[[ "$N_GPUS" -gt 0 && "$TRAIN_BATCH_SIZE" -gt 0 && "$VAL_BATCH_SIZE" -gt 0 ]] || {
  echo "Invalid GPU or batch-size setting" >&2; exit 1;
}

if [[ "$MODE" == lora || "$MODE" == both ]]; then
  [[ -n "$CKPT" && -d "$CKPT/actor" ]] || {
    echo "For LoRA, set CKPT=/absolute/path/to/global_step_202" >&2; exit 1;
  }
  for ((rank=0; rank<N_GPUS; rank++)); do
    shard="$CKPT/actor/model_world_size_${N_GPUS}_rank_${rank}.pt"
    [[ -f "$shard" ]] || { echo "Missing model shard: $shard" >&2; exit 1; }
  done
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VERL_VLLM_DISTRIBUTED_BACKEND=zeromq
export VERL_LORA_MISSING_POLICY=error
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export RAY_prestart_worker_first_driver=0
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/${USER:-user}/agentgym_hf_datasets}"
mkdir -p "$HF_DATASETS_CACHE"
# Do not import legacy AgentGym-RL/verl from the old baseline workflow.
unset PYTHONPATH VLLM_ATTENTION_BACKEND
cd "$PROJECT"

run_eval() {
  local variant="$1" out="$OUT_ROOT/$1" lora_rank=0
  local -a resume_args=()
  if [[ "$variant" == lora ]]; then
    lora_rank="$LORA_RANK"
    resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CKPT")
  else
    resume_args=(trainer.resume_mode=disable trainer.resume_from_path=null)
  fi

  [[ ! -e "$out/eval.log" ]] || {
    echo "Evaluation already exists: $out/eval.log. Change EVAL_NAME to avoid overwrite." >&2
    return 1
  }
  mkdir -p "$out/trajectories"
  echo "[BabyAI eval] variant=$variant val=$VAL_FILE out=$out lora_rank=$lora_rank"
  RAY_ADDRESS=local python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "data.train_files=$TRAIN_FILE" \
    "data.val_files=$VAL_FILE" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.val_batch_size=$VAL_BATCH_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    "actor_rollout_ref.model.path=$MODEL" \
    "actor_rollout_ref.model.lora_rank=$lora_rank" \
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_CONFIG" \
    actor_rollout_ref.rollout.agent.custom_async_server.path=pkg://agentgym_adapter.vllm_compat \
    actor_rollout_ref.rollout.agent.custom_async_server.name=AgentGymAsyncvLLMServer \
    reward_model.enable=False \
    "custom_reward_function.path=$PROJECT/agentgym_adapter/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.nnodes=1 \
    "trainer.n_gpus_per_node=$N_GPUS" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=null \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "trainer.validation_data_dir=$out/trajectories" \
    "trainer.default_local_dir=$out/checkpoints" \
    "trainer.experiment_name=eval_${EVAL_NAME}_${variant}" \
    trainer.project_name=agentgym_verl \
    "${resume_args[@]}" \
    trainer.logger='[console]' \
    ray_init.num_cpus=16 \
    2>&1 | tee "$out/eval.log"
}

case "$MODE" in
  base) run_eval base ;;
  lora) run_eval lora ;;
  both) run_eval base; run_eval lora ;;
esac

echo "[BabyAI eval] Finished: $OUT_ROOT"
