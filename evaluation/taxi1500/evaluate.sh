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

#MODEL=${1:-cis-lmu/glot500-base}
MODEL="roberta-base"
MODEL_TYPE="roberta"

NUM_PRIMITIVE=400

USE_INITIALIZATION="true"
CHECKPOINT_NUM=0
RANDOM_INITIALIZATION="false"

DATA_DIR="/dss/dsshome1/0B/ra32qov2/datasets/retrieval_bible_test/"
OUTPUT_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/evaluation/taxi1500/taxi_results/"
EMBEDDING_DIR="/dss/dsshome1/0B/ra32qov2/hyper-ofa/outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/2024-12-17_11-13-07/hyperofa_rob_all_400"


python -u evaluate.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --only_eng_vocab "false" \
    --use_initialization $USE_INITIALIZATION \
    --random_initialization $RANDOM_INITIALIZATION \
    --checkpoint_num $CHECKPOINT_NUM \
    --num_primitive $NUM_PRIMITIVE \
    --init_checkpoint 0 \
    --embedding_dir $EMBEDDING_DIR
