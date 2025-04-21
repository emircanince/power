export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

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


# export CUDA_VISIBLE_DEVICES=0

# model_name=TimeXer
# des='Timexer-MS'
# patch_len=24


# python3 -u run.py \
#   --is_training 1 \
#   --task_name long_term_forecast \
#   --root_path ./data/ \
#   --data_path forecast_data.csv \
#   --model_id EPF_168_24 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 168 \
#   --pred_len 24 \
#   --e_layers 3 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 1 \
#   --des $des \
#   --patch_len $patch_len \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 4 \
#   --itr 1

# export CUDA_VISIBLE_DEVICES=0

# model_name=iTransformer_OLD

# python -u run.py \
#   --is_training 1 \
#   --task_name long_term_forecast \
#   --root_path ./data/ \
#   --data_path forecast_data.csv \
#   --model_id ECL_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.0005 \
#   --itr 1