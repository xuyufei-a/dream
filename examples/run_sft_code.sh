#!/bin/bash
#SBATCH -J base                   # 作业名为 test
#SBATCH -o dream_cpt_baseline_1B.out
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx               # 作业使用的 QoS 为 lv0b
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH -t 30:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:4              # 单个节点使用 1 块 GPU 卡

# source ~/.conda/etc/profile.d/conda.sh
source ~/.bashrc
conda activate  dreamon

set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: examples/run_sft_tulu3.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

nproc_per_node=4
save_path=outputs

# Shift the arguments so $@ refers to the rest
shift 2

export WANDB_API_KEY=752ba61bc4d8eab9818676ef82b13bc57f4d3105
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_cpt_trainer \
    diffusion.time_reweighting=linear \
    data.train_files=../datasets/opencoder-annealing-corpus/train_data.parquet \
    data.val_files=../datasets/opencoder-annealing-corpus/eval_data.parquet \
    data.max_length=1024 \
    data.truncation=right \
    optim.lr=2e-6 \
    data.micro_batch_size_per_gpu=8 \
    +data.enable_perbatch_cutoff=False \
    data.perbatch_cutoff=False \
    model.partial_pretrain=../huggingface/Dream-Coder-v0-Base-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=test_exp \
    trainer.project_name=diff-verl-pt \
    trainer.experiment_name=test_exp \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.total_training_steps=4000 \
    2>&1
    # trainer.save_checkpoint_steps=1 \
    # data.perbatch_cutoff_type=random_with_input_pad \
    # data.train_files=../datasets/opencoder-annealing-corpus/train_data.parquet \
    # data.val_files=../datasets/opencoder-annealing-corpus/eval_data.parquet \