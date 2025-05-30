#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

MODEL=TimeXer
MODEL_ID=EPF_168_24
DATA=custom
ROOT=./data
CSV=causal_data.csv

LR_LIST=(1e-3 5e-4 1e-4)
DO_LIST=(0.0 0.1 0.2)
EL_LIST=(2 3 4)
DM_LIST=(256 512)
PL_LIST=(16 24 48)

declare -A FEAT PRED ENCDEC TAG
# ---------- setup definitions ----------
FEAT[S]="S"         ; PRED[S]="__NONE__"                           ; ENCDEC[S]=1 ; TAG[S]="S"
FEAT[MS_Pen]="MS"   ; PRED[MS_Pen]="solar_penetration,wind_penetration" ; ENCDEC[MS_Pen]=3 ; TAG[MS_Pen]="MS-Pen"
FEAT[MS_Raw]="MS"   ; PRED[MS_Raw]="solar_forecast,wind_forecast,total_load" ; ENCDEC[MS_Raw]=4 ; TAG[MS_Raw]="MS-Raw"
# ---------------------------------------

for SETUP in S MS_Pen MS_Raw; do
  LOG=tuning_${SETUP}.log ; : > "$LOG"
  echo "===== SWEEP for ${SETUP} =====" | tee -a "$LOG"

  COMMON="--is_training 1 \
    --task_name long_term_forecast \
    --model_id ${MODEL_ID}_${SETUP} \
    --model ${MODEL} \
    --data ${DATA} \
    --root_path ${ROOT} \
    --data_path ${CSV} \
    --features ${FEAT[$SETUP]} \
    --seq_len 168 --label_len 48 --pred_len 24 \
    --enc_in ${ENCDEC[$SETUP]} --dec_in ${ENCDEC[$SETUP]} --c_out 1 \
    --batch_size 32 --train_epochs 20 \
    --des tune-${TAG[$SETUP]} \
    --predictor ${PRED[$SETUP]}"          # always add flag; dummy for S

  for LR in "${LR_LIST[@]}"; do
    for DO in "${DO_LIST[@]}"; do
      for EL in "${EL_LIST[@]}"; do
        for DM in "${DM_LIST[@]}"; do
          for PL in "${PL_LIST[@]}"; do
            echo ">>> ${SETUP} LR=${LR} DO=${DO} EL=${EL} DM=${DM} PL=${PL}" | tee -a "$LOG"
            python run.py ${COMMON} \
              --learning_rate ${LR} \
              --dropout ${DO} \
              --e_layers ${EL} \
              --d_model ${DM} \
              --patch_len ${PL} \
            | tee -a "$LOG"
          done
        done
      done
    done
  done
done