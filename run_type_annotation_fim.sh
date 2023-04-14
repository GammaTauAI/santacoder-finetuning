#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_type_annotation_fim.sh <num_gpus> <batch_size> <num_workers>"
    exit 1
fi

#CUDA_VISIBLE_DEVICES=... python3 -m torch.distributed.launch \
python3 -m torch.distributed.launch \
        --nproc_per_node $1 train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="nuprl/ts-training" \
        --output_dir="./chk" \
        --subset="data" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 1000000 \
        --batch_size $2 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --num_warmup_steps 100 \
        --eval_freq 500 \
        --save_freq 5000 \
        --fim_rate 1 \
        --fim_spm_rate 0.5 \
        --streaming \
        --log_freq 1 \
        --num_workers="$3" \
        --no_fp16 \
        --bf16 \
        --hub_model_id="gammatau/santacoder-ts-fim" \
        --push_to_hub \
        #--checkpoint "chk/last-checkpoint"
