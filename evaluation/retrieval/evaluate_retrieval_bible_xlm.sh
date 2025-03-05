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

MODEL="xlm-roberta-base"
MODEL_TYPE="xlmr"

MAX_LENGTH=512
LC=""
BATCH_SIZE=128
DIM=768
NUM_PRIMITIVE=400
NLAYER=12
LAYER=7

# set use_initialization "true" to use models with initialization
USE_INITIALIZATION="true"
# set checkpoint_num=0 to use models without continue pretraining
# CHECKPOINT_NUM=4000
CHECKPOINT_NUM=0

# set random_initialization "true" to use models with random initialization for embeddings of new words
RANDOM_INITIALIZATION="false"

# paths
DATA_DIR="/dss/dsshome1/0B/ra32qov2/datasets/retrieval_bible_test/"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/retrieval/bible/"
TOKENIZED_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/retrieval/bible_tokenized_xlm_r"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/2025-03-04_21-55-02/hyperofa_xlm_all_400"
# INIT_CHECKPOINT="/dss/dsshome1/0B/ra32qov2/hyper-ofa/continued_pretraining/outputs/_random_multi_400/checkpoint-$CHECKPOINT_NUM"
INIT_CHECKPOINT=0

python -u evaluation/retrieval/evaluate_retrieval_bible.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --embed_size $DIM \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_LENGTH \
    --num_layers $NLAYER \
    --dist cosine $LC \
    --specific_layer $LAYER \
    --only_eng_vocab "false" \
    --use_initialization $USE_INITIALIZATION \
    --random_initialization $RANDOM_INITIALIZATION \
    --num_primitive $NUM_PRIMITIVE \
    --tokenized_dir $TOKENIZED_DIR \
    --checkpoint_num $CHECKPOINT_NUM \
    --init_checkpoint $INIT_CHECKPOINT \
    --embedding_dir $EMBEDDING_DIR
