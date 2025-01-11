# Model Parameters
MODEL="roberta-base"
NUM_PRIMITIVE=400

# Training Parameters
PER_DEVICE_TRAIN_BATCH_SIZE=12
GRAD_ACC_STEPS=16
EPOCHS=100
SAVE_STEPS=10000

# Paths
TRAIN_DATA_DIR="/mounts/data/proj/ayyoobbig/1000LM/data/1000LM.txt"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/continued_pretraining/"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/2025-01-11_00-21-58/hyperofa_rob_all_400"

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ../run_extra.py \
  --model_name_or_path $MODEL \
  --train_file $TRAIN_DATA_DIR \
  --tokenizer_name /mounts/data/proj/ayyoobbig/1000LM/tokenizer/1000LM_extended_spm \
  --output_dir $OUTPUT_DIR \
  --cache_dir /mounts/data/proj/ayyoobbig/ofa/cache \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --fp16 True \
  --do_train \
  --num_train_epochs $EPOCHS \
  --save_steps $SAVE_STEPS \
  --ddp_timeout 259200 \
  --use_initialization True \
  --random_initialization False \
  --num_primitive $NUM_PRIMITIVE \
  --embedding_dir $EMBEDDING_DIR \
  --only_eng_vocab False \
  --preprocessing_num_workers 8 \
  --ignore_data_skip True
