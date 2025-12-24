set -ex
export CUDA_VISIBLE_DEVICES=4,5

############## SimpleRL ##############
MODEL_NAME_OR_PATH_LIST=(
    Qwen/Qwen2.5-MATH-7B
)

PROMPT_TYPE="qwen25-math-cot"

DATA_NAME="gsm8k,math500,olympiadbench"

for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATH_LIST[@]}"
do
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name $DATA_NAME \
        --output_dir "score_pass_1/${MODEL_NAME_OR_PATH}" \
        --prompt_type $PROMPT_TYPE \
        --num_test_sample "-1" \
        --temperature 0.7 \
        --n_sampling 4 \
        --top_p 1 \
        --use_vllm \
        --apply_chat_template \
        --save_outputs
done

DATA_NAME="amc23,aime24,aime25"
for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATH_LIST[@]}"
do
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name $DATA_NAME \
        --output_dir "score_pass_1/${MODEL_NAME_OR_PATH}" \
        --prompt_type $PROMPT_TYPE \
        --num_test_sample "-1" \
        --temperature 0.7 \
        --n_sampling 64 \
        --top_p 1 \
        --use_vllm \
        --apply_chat_template \
        --save_outputs
done