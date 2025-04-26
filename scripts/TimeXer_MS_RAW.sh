#!/usr/bin/env bash
# TimeXer – MS with raw forecasts + load
export CUDA_VISIBLE_DEVICES=0

python3 -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path causal_data.csv \
  --model_id EPF_168_24 \
  --model TimeXer \
  --data custom \
  --features MS \
  --predictor solar_forecast,wind_forecast,total_load \
  --seq_len 168 --label_len 48 --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --patch_len 24 \
  --d_model 512 --d_ff 512 --e_layers 3 \
  --batch_size 32 --itr 1 \
  --des TimeXer-MS-Raw \
  "$@"