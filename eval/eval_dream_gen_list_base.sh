
tasks="gsm8k_cot mbpp minerva_math bbh"
nshots="8 3 4 3"
lengths="256 512 512 512"
temperatures="0 0.2 0 0"
models="/data5/xyf/huggingface/Dream-v0-Base-7B"

# model=/data5/xyf/huggingface/Dream-Coder-v0-Base-7B
# model='/data5/xyf/Dream/outputs/dream_cpt_baseline/global_step_100'
# model='/data5/xyf/Dream/outputs/dream_cpt_adapt/global_step_100'
# Create arrays from space-separated strings
read -ra MODELS_ARRAY <<< "$models"
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"

export HF_ALLOW_CODE_EVAL=1

for model in "${MODELS_ARRAY[@]}"; do
    killall python3.10
    echo "Evaluating model: $model"

    accelerate launch --main_process_port 29510 eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=512,diffusion_steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path evals_results/humaneval-ns0 \
        --log_samples \
        --confirm_run_unsafe_code 
    ### NOTICE: use postprocess for humaneval
    # python postprocess_code.py {the samples_xxx.jsonl file under output_path}

    # Iterate through the arrays
    killall python3.10
    for i in "${!TASKS_ARRAY[@]}"; do
        output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
        echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
        accelerate launch eval.py --model dream \
            --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95 \
            --tasks ${TASKS_ARRAY[$i]} \
            --num_fewshot ${NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code
    done
done
