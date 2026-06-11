torchrun --nproc_per_node=8 ueval_klein.py \
    --planner_path PATH_TO_PLANNER_CKPT \
    --critic_path PATH_TO_CRITIC_CKPT \
    --output_path result/test.json \
    --output_image_dir result/images \
    --max_step_iterations 3