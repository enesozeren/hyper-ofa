#!/bin/bash


source_model_name="roberta-base"
target_model_name="cis-lmu/glot500-base"
source_language_set="None"
target_language_set="None"
keep_dim="[400]"

python -u ofa/ofa_initialize.py \
    --source_model_name $source_model_name \
    --target_model_name $target_model_name \
    --source_language_set $source_language_set \
    --target_language_set $target_language_set \
    --keep_dim $keep_dim \
    --do_save "true"
