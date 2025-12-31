set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: examples/run_sft_tulu3.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

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
    model.partial_pretrain=../huggingface/Dream-v0-Base-7B-5l \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=test_exp \
    trainer.project_name=diff-verl-pt \
    trainer.experiment_name=test_exp \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.total_training_steps=4000 \
    # trainer.save_checkpoint_steps=1 \
    # data.perbatch_cutoff_type=random_with_input_pad \
    # data.train_files=../datasets/opencoder-annealing-corpus/train_data.parquet \
    # data.val_files=../datasets/opencoder-annealing-corpus/eval_data.parquet \