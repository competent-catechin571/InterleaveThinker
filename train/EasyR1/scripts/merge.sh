#!/usr/bin/env bash
source activate

ROOT=ROOT/ckpt/critic_rl

for d in "$ROOT"/global_step_*/actor; do
  echo "Merging $d ..."
  python ROOT/code/InterleaveThinker/train/EasyR1/scripts/model_merger.py --local_dir "$d"
done
