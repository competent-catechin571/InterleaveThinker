export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN

ROOT=
export HOST_TAG="$(hostname | tr -cd '[:alnum:]-_')"
export BASE_CACHE="${ROOT}/.cache/${HOST_TAG}"
export XDG_CACHE_HOME="${BASE_CACHE}/.cache"   
export TORCH_CUDA_KERNEL_CACHE_PATH="${BASE_CACHE}/.cache/torch/kernels"
export TORCH_EXTENSIONS_DIR="${BASE_CACHE}/torch_extensions"
export TRITON_CACHE_DIR="${BASE_CACHE}/triton"

mkdir -p "$TORCH_CUDA_KERNEL_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

export HF_DATASETS_CACHE="${ROOT}"

RUN_NAME=sft_critic

source activate ${ROOT}/envs/llamafactory

PROJECT_PATH="${ROOT}/code/InterleaveThinker"

eval $(python $PROJECT_PATH/train/LLaMA-Factory/local_scripts/hope_deepspeed_distributed_launch.py)

echo $FORCE_TORCHRUN
echo $NNODES
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT

CONFIG_YAML="$PROJECT_PATH/train/LLaMA-Factory/examples/train_full/sft_critic.yaml"

cd $PROJECT_PATH/train/LLaMA-Factory

llamafactory-cli train $CONFIG_YAML


