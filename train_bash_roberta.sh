# Model Parameters
MODEL="roberta-base"
NUM_PRIMITIVE=400

# Training Parameters
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRAD_ACC_STEPS=4
EPOCHS=1
SAVE_STEPS=5000
LOGGING_STEPS=1000

# Paths
TRAIN_DATA_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/continued_pretraining/dataset/hyperofa_training_data.txt"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/continued_pretraining/outputs/"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/2025-01-11_00-21-58/hyperofa_rob_all_400"

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_extra.py \
  --model_name_or_path $MODEL \
  --train_file $TRAIN_DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --cache_dir /dss/dsshome1/0B/ra32qov2/hyper-ofa/caches \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --fp16 True \
  --do_train \
  --num_train_epochs $EPOCHS \
  --save_steps $SAVE_STEPS \
  --logging_steps $LOGGING_STEPS \
  --ddp_timeout 259200 \
  --use_initialization True \
  --random_initialization False \
  --num_primitive $NUM_PRIMITIVE \
  --embedding_dir $EMBEDDING_DIR \
  --only_eng_vocab False \
  --preprocessing_num_workers 8 \
  --ignore_data_skip True
