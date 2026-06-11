#!/usr/bin/env bash
source activate

ROOT=ROOT/ckpt/critic_rl

TARGET_DIR="$ROOT/global_step_xxx/actor"

echo "Merging $TARGET_DIR ..."
python ROOT/code/InterleaveThinker/train/EasyR1/scripts/model_merger.py --local_dir "$TARGET_DIR"