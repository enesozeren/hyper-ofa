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
NLAYER=12
LAYER=7
NUM_PRIMITIVE=100
# set use_initialization "true" to use models with initialization
USE_INITIALIZATION="true"
# set checkpoint_num=0 to use models without continue pretraining
CHECKPOINT_NUM=0
# set random_initialization "true" to use models with random initialization for embeddings of new words
RANDOM_INITIALIZATION="false"

# set checkpoint_num=0 to use models without continue pretraining

# set use_initialization "true" to use models with initialization

# set random_initialization "true" to use models with random initialization for embeddings of new words

# only comment init_checkpoint (i.e., set it None) if you want to the huggingface pretrained model, e.g., roberta
# otherwise always keep it uncommented

# paths
DATA_DIR="/dss/dsshome1/0B/ra32qov2/datasets/retrieval_tatoeba/"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/retrieval/tatoeba/"
TOKENIZED_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/retrieval/tatoeba_tokenized_xlm_r"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/ofa_multi_100"

python -u evaluation/retrieval/evaluate_retrieval_tatoeba.py \
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
    --checkpoint_num $CHECKPOINT_NUM \
    --num_primitive $NUM_PRIMITIVE \
    --tokenized_dir $TOKENIZED_DIR \
    --init_checkpoint 0 \
    --embedding_dir $EMBEDDING_DIR

