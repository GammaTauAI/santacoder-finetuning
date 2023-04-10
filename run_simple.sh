#!/bin/bash
python -m torch.distributed.launch \
        --nproc_per_node 3 train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
  --output_dir="./chk2" \
        --subset="data/typescript" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 100 \
        --eval_freq 100 \
        --save_freq 100 \
  --fim_rate 0.5 \
  --fim_spm_rate 0.5 \
  --streaming \
        --log_freq 1 \
        --num_workers="20" \
  --no_fp16 \
  --bf16

