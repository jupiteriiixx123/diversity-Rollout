#!/usr/bin/env bash
set -Eeuo pipefail

# ===== 每次实验只需要改这里 =====
RUN_NAME="${RUN_NAME:-babyai_epoch3_v3}"
MODE="${MODE:-fresh}"                  # fresh | resume
RESUME_FROM="${RESUME_FROM:-}"         # resume 时：已有 global_step_XXX 目录
EPOCHS="${EPOCHS:-3}"
STEPS="${STEPS:-auto}"                 # auto | 正整数
GPU_IDS="${GPU_IDS:-0,2,3,4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
SAVE_EVERY="${SAVE_EVERY:-30}"
# ================================

ROOT="${AGENTGYM_ROOT:-/public/home/hh_hh/AgentGym-RL}"
P="$ROOT/agentgym-verl-adapter"
M="$ROOT/models/Qwen2.5-7B-Instruct"
D="$P/data/babyai_train.parquet"
A="$P/configs/babyai_one_step_agent.yaml"
OUT="$P/runs/$RUN_NAME"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"; NG=${#GPU_ARRAY[@]}

[[ -f "$D" && -f "$A" && -d "$M" ]] || { echo 'ERROR: 模型、数据或 Agent 配置不存在' >&2; exit 1; }
[[ "$NG" -gt 0 && "$BATCH_SIZE" -gt 0 && "$ROLLOUT_N" -gt 1 ]] || { echo 'ERROR: GPU/Batch/Rollout 参数不合法' >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: EPOCHS 必须是正整数' >&2; exit 1; }
[[ "$STEPS" == auto || "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: STEPS 必须是 auto 或正整数' >&2; exit 1; }

if [[ "$MODE" == fresh ]]; then
  [[ ! -e "$OUT/train.log" && ! -e "$OUT/checkpoints/latest_checkpointed_iteration.txt" ]] || {
    echo "ERROR: 已存在同名实验：$OUT；请修改 RUN_NAME" >&2; exit 1;
  }
  RESUME_ARGS=(trainer.resume_mode=disable)
elif [[ "$MODE" == resume ]]; then
  [[ -n "$RESUME_FROM" && -d "$RESUME_FROM/actor" ]] || { echo 'ERROR: RESUME_FROM 路径无效' >&2; exit 1; }
  for ((i=0; i<NG; i++)); do
    file="$RESUME_FROM/actor/model_world_size_${NG}_rank_${i}.pt"
    [[ -f "$file" ]] || { echo "ERROR: 缺少 Checkpoint 分片：$file" >&2; exit 1; }
  done
  RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$RESUME_FROM")
else
  echo 'ERROR: MODE 只能是 fresh 或 resume' >&2; exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS" VLLM_USE_V1=1 VERL_VLLM_DISTRIBUTED_BACKEND=zeromq
export VERL_LORA_MISSING_POLICY=error RAY_prestart_worker_first_driver=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/${USER:-user}/agentgym_hf_datasets}"
unset PYTHONPATH VLLM_ATTENTION_BACKEND
mkdir -p "$OUT" "$HF_DATASETS_CACHE"; cd "$P"

ARGS=(
  algorithm.adv_estimator=grpo
  "data.train_files=$D" "data.val_files=$D" "data.train_batch_size=$BATCH_SIZE"
  data.max_prompt_length=1024 data.max_response_length=4096
  data.return_raw_chat=True data.dataloader_num_workers=0
  "actor_rollout_ref.model.path=$M"
  "actor_rollout_ref.model.lora_rank=$LORA_RANK" "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.rollout.mode=async "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$NG"
  actor_rollout_ref.rollout.max_model_len=5120 actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  "actor_rollout_ref.rollout.agent.num_workers=$BATCH_SIZE"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$A"
  actor_rollout_ref.rollout.agent.custom_async_server.path=pkg://agentgym_adapter.vllm_compat
  actor_rollout_ref.rollout.agent.custom_async_server.name=AgentGymAsyncvLLMServer
  "custom_reward_function.path=$P/agentgym_adapter/reward.py"
  "trainer.n_gpus_per_node=$NG" "trainer.total_epochs=$EPOCHS" "trainer.save_freq=$SAVE_EVERY" "trainer.max_actor_ckpt_to_keep=3"
  trainer.val_before_train=False
  "trainer.default_local_dir=$OUT/checkpoints" "trainer.rollout_data_dir=$OUT/rollouts"
  trainer.logger='[console]' trainer.project_name=agentgym_verl "trainer.experiment_name=$RUN_NAME"
  ray_init.num_cpus=16 "${RESUME_ARGS[@]}"
)
[[ "$STEPS" == auto ]] && ARGS+=(trainer.total_training_steps=null) || ARGS+=("trainer.total_training_steps=$STEPS")

echo "[TRAIN] run=$RUN_NAME mode=$MODE GPUs=$GPU_IDS batch=$BATCH_SIZE n=$ROLLOUT_N"
echo "[TRAIN] epochs=$EPOCHS steps=$STEPS resume=$RESUME_FROM outputs=$OUT"
RAY_ADDRESS=local python -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"