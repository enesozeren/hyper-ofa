DATE="2025-01-03_01-50-54"
CHECKPOINT="model-epoch=119.ckpt"

python ofa/mapping_model_inference.py \
--test_or_inference inference \
--setformer_config_path setformer/configs/setformer_config.yaml \
--test_inference_mapping_data_path outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data/target_subword_to_word_mapping.pkl \
--checkpoint_path outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-100/setformer_training_logs/$DATE/checkpoints/$CHECKPOINT \
--keep_dim 100

python ofa/init_target_matrix.py \
--source_matrix_path outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data/source_matrix.npy \
--source_model_name xlm-roberta-base \
--setformer_predictions_path outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-100/setformer_training_logs/$DATE/inference_logs/prediction_dict.pkl

bash evaluation/retrieval/evaluate_retrieval_tatoeba_xlm.sh