#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Model parameters
MODEL="xlm-roberta-base"
MODEL_TYPE="xlmr"

NUM_PRIMITIVE=200
# set checkpoint_num=0 to use models without continue pretraining
CHECKPOINT_NUM=0
# set random_initialization "true" to use models with random initialization for embeddings of new words
RANDOM_INITIALIZATION="false"
# set use_initialization "true" to use models with initialization
USE_INITIALIZATION="true"

# Finetuning parameters
NUM_EPOCHS=10
LR=2e-5
LC=""
BATCH_SIZE=32
GRAD_ACC=1
MAX_LENGTH=256

# Paths
DATA_DIR="/dss/dsshome1/0B/ra32qov2/datasets/pos/"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/tagging/pos/"
TOKENIZED_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/tagging/pos_tokenized"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/random_xlm_all_200"

python -u evaluation/tagging/evaluate_pos.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --output_dir $OUTPUT_DIR \
    --max_seq_len $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size 32 \
    --save_steps 500 \
    --seed 1 \
    --learning_rate $LR \
    --do_train \
    --do_eval \
    --do_predict \
    --train_langs eng_Latn \
    --eval_all_checkpoints \
    --eval_patience -1 \
    --overwrite_output_dir \
    --save_only_best_checkpoint $LC \
    --only_eng_vocab "false" \
    --use_initialization $USE_INITIALIZATION \
    --random_initialization $RANDOM_INITIALIZATION \
    --checkpoint_num $CHECKPOINT_NUM \
    --num_primitive $NUM_PRIMITIVE \
    --tokenized_dir $TOKENIZED_DIR \
    --init_checkpoint 0 \
    --embedding_dir $EMBEDDING_DIR
