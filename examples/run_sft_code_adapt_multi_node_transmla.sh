#!/bin/bash
#SBATCH -J base                   # 作业名为 test
#SBATCH -o dream_cpt_adapt.out
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

MASTER_NODE=$1
NUM_NODES=$2
NODE_RANK=$3

LR=1e-5
BATCH_SIZE=1024
STEP=2000
lr_scheduler=cosine
time_reweighting=linear
use_focal_loss=false
noise_dist=uniform
enable_cutoff=true
weight_eos=true

dataset_name=transmla+math
base_model_path=../huggingface/Dream-v0-Base-7B
save_path=outputs/dream_${dataset_name}_${time_reweighting}_use_focal_${use_focal_loss}_noise_dist_${noise_dist}_enable_cutoff_${enable_cutoff}_cpt_adapt_bs${BATCH_SIZE}_step${STEP}_${lr_scheduler}_lr${LR}_weight_eos_${weight_eos}
# save_path=outputs/dream_test
# rm /data/muhan/.cache/huggingface -r
exp_name=$(basename $save_path)
killall python3.10

# # Shift the arguments so $@ refers to the rest
# shift 2

JOB_ID=$(basename $save_path)
export WANDB_API_KEY=752ba61bc4d8eab9818676ef82b13bc57f4d3105
torchrun \
    --rdzv-endpoint=$MASTER_NODE:8888 \
    --rdzv-id=$JOB_ID \
    --nnodes=$NUM_NODES \
    --nproc-per-node=8 \
    --node_rank=$NODE_RANK \
    -m src.trainer.fsdp_cpt_trainer \
    diffusion.time_reweighting=${time_reweighting} \
    data.train_files=../datasets/${dataset_name}/train_data.parquet \
    data.val_files=../datasets/${dataset_name}/eval_data.parquet \
    data.max_length=1024 \
    data.truncation=right \
    optim.lr=$LR \
    optim.lr_scheduler=$lr_scheduler \
    data.micro_batch_size_per_gpu=8 \
    data.train_batch_size=$BATCH_SIZE \
    +data.enable_perbatch_cutoff=${enable_cutoff} \
    data.perbatch_cutoff=${enable_cutoff} \
    model.partial_pretrain=${base_model_path} \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    model.mask_free=True \
    trainer.project_name=diff-verl-pt \
    trainer.experiment_name=$exp_name \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=$save_path \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEP \
    trainer.save_checkpoint_steps=3000 \
    diffusion.token_reweighting=${use_focal_loss} \
    diffusion.noise_level_distribution=${noise_dist} \
    diffusion.weight_eos=${weight_eos} \
    2>&1
    # trainer.save_checkpoint_steps=1 \
    # data.perbatch_cutoff_type=random_with_input_pad \
    # data.train_files=../datasets/opencoder-annealing-corpus/train_data.parquet \
    # data.val_files=../datasets/opencoder-annealing-corpus/eval_data.parquet \