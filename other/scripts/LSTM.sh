# #!/usr/bin/env bash
# # ===============================
# # Train / evaluate the LSTM model
# # ===============================

# # Pick the GPU(s) you want to use
# export CUDA_VISIBLE_DEVICES=0

# # Name of the model file/class you just added
# model_name=LSTM

# python -u run.py \
#   --is_training 1 \
#   --task_name long_term_forecast \
#   --root_path ./data/ \
#   --data_path causal_data.csv \
#   --model_id LSTM_168_24 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --predictor solar_forecast,wind_forecast,total_load \
#   --seq_len 168 \
#   --pred_len 24 \
#   \
#   # --- LSTM-specific hyper-parameters -----------------
#   --lstm_hidden 128          \  # hidden size
#   --lstm_num_layers 2        \  # number of stacked LSTM layers
#   --lstm_bidirectional 1     \  # 1 = bidirectional, 0 = unidirectional
#   \
#   # --- IO dimensions (match your dataset columns) -----
#   --enc_in 4   \  # input size
#   --dec_in 4   \  # decoder input size (if different, change accordingly)
#   --c_out 1    \  # output size
#   \
#   --des 'Exp' \
#   --batch_size 4 \
#   --learning_rate 0.0005 \
#   --itr 1 \
#   "$@"

export CUDA_VISIBLE_DEVICES=0

model_name=LSTM

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path causal_data.csv \
  --model_id ECL_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --predictor solar_forecast,wind_forecast,total_load \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --itr 1 \
  "$@"    