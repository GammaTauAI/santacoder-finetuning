#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_type_annotation_fim.sh <num_gpus> <batch_size> <num_workers>"
    exit 1
fi

python -m torch.distributed.launch \
        --nproc_per_node $1 train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="nuprl/ts-training" \
        --output_dir="./chk" \
        --subset="data" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size $2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 100 \
        --eval_freq 100 \
        --save_freq 100 \
        --fim_rate 0.5 \
        --fim_spm_rate 0.5 \
        --streaming \
        --log_freq 1 \
        --num_workers="$3" \
        --no_fp16 \
        --bf16 \
        --hub_model_id="gammatau/santacoder-ts-fim" \
        # --checkpoint "chk/last-checkpoint" \
        --push_to_hub True \

