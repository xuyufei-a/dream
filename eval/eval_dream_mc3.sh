# tasks="mmlu arc_easy arc_challenge hellaswag piqa gpqa_main_n_shot winogrande race"
# nshots="5 0 0 0 0 5 5 0"
tasks="arc_easy arc_challenge hellaswag winogrande race"
nshots="0 0 0 5 0"
# tasks="arc_easy arc_challenge hellaswag piqa gpqa_main_n_shot winogrande race"
# nshots="0 0 0 0 5 5 0"
# tasks="arc_easy arc_challenge piqa winogrande"
# nshots="0 0 0 5"
# tasks="mmlu"
# nshots="5"

# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
model=/data5/xyf/Dream/outputs/dream_transmla+math_cart_weight-eos-false_cpt_baseline_bs1024_step2000_cosine_lr1e-5/global_step_2000
mask_free=False

export HF_TOKEN=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
export HF_ENDPOINT=https://hf-mirror.com
# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    killall python3.10
    output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    accelerate launch --main_process_port 29510 eval.py --model dream \
        --model_args pretrained=${model},add_bos_token=true,dtype=bfloat16,mask_free=${mask_free} \
        --tasks ${TASKS_ARRAY[$i]} \
        --batch_size 8 \
        --output_path $output_path \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --log_samples \
        --confirm_run_unsafe_code 
done
