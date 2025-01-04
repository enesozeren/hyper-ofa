DATE="2025-01-04_23-16-46"
CHECKPOINT="model-epoch=79.ckpt"

python ofa/mapping_model_inference.py \
--test_or_inference inference \
--hypernetwork_config_path hypernetwork/configs/hypernetwork_config.yaml \
--test_inference_mapping_data_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/mapping_data/target_subword_to_word_mapping.pkl \
--checkpoint_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/$DATE/checkpoints/$CHECKPOINT \
--keep_dim 400

python ofa/init_target_matrix.py \
--source_matrix_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/mapping_data/source_matrix.npy \
--source_model_name roberta-base \
--hypernetwork_predictions_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-400/hypernetwork_training_logs/$DATE/inference_logs/prediction_dict.pkl

bash evaluation/retrieval/evaluate_retrieval_tatoeba_roberta.sh