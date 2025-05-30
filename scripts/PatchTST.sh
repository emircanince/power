export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path thesis_data.csv \
  --model_id PatchTST_168_24 \
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