#!/bin/bash

# ==========================================
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=7
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export DECORD_EOF_RETRY_MAX=2048001
export NCCL_NET=Socket

export WANDB_CONSOLE=off
export TQDM_USE_WRITELN=0
export VLLM_USE_V1=1

export OPENAI_API_KEY=YOUR_API_KEY
export EDIT_API_ENDPOINT=YOUR_SERVER_IP
export EDIT_MODEL_NAME="klein"
export EDIT_API_MAX_WORKERS=128
export MAX_DURATION=70

set -x
set -e
set -o pipefail

ROOT=
source /usr/local/conda/bin/activate ${ROOT}/envs/easyr1-qwen3

# ==========================================
# 2. Ray start
# ==========================================
echo "Cleaning up stale Ray processes..."
ray stop --force || true
pkill -9 -u $USER -f "raylet|gcs_server|plasma_store|ray_worker" || true
sleep 2

ACTUAL_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)
if [ -n "$ACTUAL_IFNAME" ]; then
    export NCCL_SOCKET_IFNAME=$ACTUAL_IFNAME
else
    unset NCCL_SOCKET_IFNAME
fi

export RAY_TMPDIR=/tmp/ray_${USER}_${RANDOM}
mkdir -p $RAY_TMPDIR

RAY_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Starting Ray on port $RAY_PORT with temp dir $RAY_TMPDIR..."

ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --temp-dir=$RAY_TMPDIR
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

# ==========================================
# 3. train
# ==========================================
PROJECT_PATH=${ROOT}/code/InterleaveThinker
WORKING_DIR=${PROJECT_PATH}/train/EasyR1
cd $WORKING_DIR

SAVE_DIR=${ROOT}/ckpt/critic_rl
IMAGE_DIR=${ROOT}/data
MODEL_PATH=${ROOT}/ckpt/critic_sft
TRAIN_FILE=${ROOT}/data/InterleaveThinker/Train-Data/critic_rl.jsonl
TEST_FILE=$TRAIN_FILE

ROLLOUT_BS=16
GLOBAL_BS=8
MB_PER_UPDATE=1
MB_PER_EXP=1
TP_SIZE=4
N_GPUS_PER_NODE=8
NNODES=1

export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH:-}"

python3 -m verl.trainer.main \
  config=${WORKING_DIR}/examples/interleave_thinker_config.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.image_dir="${IMAGE_DIR}" \
  data.rollout_batch_size="${ROLLOUT_BS}" \
  data.answer_key="evaluation" \
  worker.actor.global_batch_size="${GLOBAL_BS}" \
  worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.strategy=adamw_bf16 \
  worker.actor.optim.lr=2e-6 \
  worker.rollout.tensor_parallel_size="${TP_SIZE}" \
  trainer.project_name="EasyR1-qwen3-vl" \
  trainer.experiment_name="inter_thinker_qwen3vl_rl_kelvin" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=50 \
  trainer.save_checkpoint_path="${SAVE_DIR}" \
  worker.reward.reward_function=${WORKING_DIR}/verl/reward_function/interleave_thinker_reward.py:compute_score