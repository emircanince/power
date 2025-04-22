#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

MODEL=TimeXer
MODEL_ID=EPF_168_24        # ← pick any identifier you like
DATA=custom
ROOT=./data
CSV=causal_data.csv

COMMON="--is_training 1 \
  --task_name long_term_forecast \
  --model_id ${MODEL_ID} \
  --model ${MODEL} \
  --data ${DATA} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features MS \
  --predictor solar_forecast,wind_forecast,total_load \
  --seq_len 168 --label_len 48 --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --des tuning"

# grid
for LR in 1e-3 5e-4 1e-4; do
  for DO in 0.0 0.1 0.2; do
    for EL in 2 3 4; do
      for DM in 256 512; do
        for PL in 16 24 48; do
          echo ">>> LR=${LR},DO=${DO},EL=${EL},DM=${DM},PL=${PL}"
          python run.py ${COMMON} \
             --learning_rate ${LR} \
             --dropout ${DO} \
             --e_layers ${EL} \
             --d_model ${DM} \
             --patch_len ${PL} \
             --batch_size 32 \
             --train_epochs 20 \
          | tee -a tuning.log
        done
      done
    done
  done
done