# HyperOFA

HyperOFA: Expanding LLM Vocabulary to New Languages via Hypernetwork Based Embedding Initialization

Abstract:
Most pretrained language models are trained primarily for high-resource languages, limiting their usability in mid and low-resource languages. A common approach to adapt these models for other languages involves introducing new tokens specific to these languages and continuing pretraining. However, the method used to initialize these newly introduced tokens significantly impacts the duration and efficiency of continued pretraining. Poor initialization can lead to longer training times and increased computational costs. OFA (Liu et al., 2023) method provides an effective initialization strategy. Building on this, HyperOFA introduces a hypernetwork-based approach to initialize new tokens. Experimental results show that HyperOFA consistently outperforms the random initialization baseline and performs on par with OFA, sometimes surpassing it and sometimes falling behind.

[Check the paper here](tbd)

## File Structure

```
.
├── README.md
├── evaluation                              <- retrieval, taggin, taxi1500 evluations
├── model_loader_extra.py
├── modeling_roberta_extra.py
├── modeling_xlmr_extra.py
├── continued_pretraining
│   └── create_training_dataset.py          <- creates dataset for continued pretraining
├── hypernetwork
│   ├── configs                             <- this folder contains a config file for hypernetwork training / inference
│   ├── dataset.py                          <- contains dataset, sampler and collate_fn for hypernetwork training
│   ├── lightning_modules.py                <- contains custom loss, lightning model class definitions
│   ├── lstm.py                             <- BiLSTM hypernetwork architecture
│   ├── setformer.py                        <- Transformer without positional encoding hypernetwork architecture
│   ├── train.py                            <- training script for the hypernetwork
│   └── utils.py                            <- util functions
├── hyperofa
│   ├── create_hypernetwork_train_data.py   <- creates the mapping from tokens to words
│   ├── train_hypernetwork.py               <- trains the hypernetwork
│   ├── inference_hypernetwork.py           <- predicts the target token embeddings with the hypernetwork
│   ├── init_target_matrix.py               <- initializes the embedding matrix with the hypernetwork
│   ├── ofa_initialize_test_set.bash        <- calculates test metrics for OFA initializations
│   ├── random_init.bash                    <- initilizes the embedding matrix randomly after copying matched ones
│   └── utils.py                            <- util functions
├── model_loader_extra.py                   <- more util functions from OFA
├── modeling_roberta_extra.py               <- custom model definitions for OFA roberta
├── modeling_xlmr_extra.py                  <- custom model definitions for OFA xlm-roberta
├── requirements.txt                        <- required python packages
├── run_extra.py                            <- continued pre-training script
├── train_bash_roberta.sh                   <- bash script for continued pre-training roberta
└── train_bash_xlm_roberta.sh               <- bash script for continued pre-training xlm-roberta
```

## Initializing Embeddings with HyperOfa

Follow these steps to initialize `roberta-base` (or `xlm-roberta-base`) model.

Step 0) You need to download the [ColexNet+](https://github.com/cisnlp/ColexificationNet) word vectors.

Step 1) Create mapping dataset introduced in [OFA](https://github.com/cisnlp/ofa)
```bash
python hyperofa/create_hypernetwork_train_data.py \
--word_vec_embedding_path colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv \
--source_model_name xlm-roberta-base \
--target_model_name cis-lmu/glot500-base \
--keep_dim 200 \
--output_dir outputs
```

Step 2) Train the hypernetwork.
```bash
python hyperofa/train_hypernetwork.py \
--word_vec_embedding_path colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv \
--keep_dim 200 \
--mapping_data_dir outputs/roberta-base_to_cis-lmu-glot500-base_dim-200/mapping_data \
--hypernetwork_config_path hypernetwork/configs/hypernetwork_config.yaml
```

Step 2.5) You can calucalte test metrics of hypernetwork to asses the quality of the embeddings predicted by the hypernetwork (Replace the test_inference_mapping_data_path and checkpoint_path arguments w.r.t. your hypernetwork training outputs from Step 2).
```bash
python hyperofa/inference_hypernetwork.py \
--test_or_inference test \
--source_matrix_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data/source_matrix.npy \
--hypernetwork_config_path hypernetwork/configs/hypernetwork_config.yaml \
--test_inference_mapping_data_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/hypernetwork_training_logs/2025-01-09_01-21-32/test_mapping_set.pkl \
--checkpoint_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/hypernetwork_training_logs/2025-01-09_01-21-32/checkpoints/model-epoch=39.ckpt \
--keep_dim 100
```

Step 3) Make inference with hypernetwork to predict the new token embeddings (Replace the test_inference_mapping_data_path, checkpoint_path arguments w.r.t. your hypernetwork training outputs from Step 2).
```bash
python hyperofa/inference_hypernetwork.py \
--test_or_inference inference \
--hypernetwork_config_path hypernetwork/configs/hypernetwork_config.yaml \
--test_inference_mapping_data_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data/target_subword_to_word_mapping.pkl \
--checkpoint_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/hypernetwork_training_logs/2025-01-08_16-35-12/checkpoints/model-epoch=119.ckpt \
--keep_dim 100
```

Step 4) Create the target matrix from the predicted embeddings from Step 3 (Replace the hypernetwork_predictions_path argument w.r.t. your outputs from Step 3). 
```bash
python hyperofa/init_target_matrix.py \
--source_matrix_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data/source_matrix.npy \
--source_model_name roberta-base \
--hypernetwork_predictions_path outputs/roberta-base_to_cis-lmu-glot500-base_dim-100/hypernetwork_training_logs/2025-01-02_13-50-08/inference_logs/prediction_dict.pkl
```

## Initializing Randomly

To create the random initialized target matrix use
```bash
python hyperofa/random_init.py \
--source_matrix_path outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-200/mapping_data/source_matrix.npy \
--source_model_name xlm-roberta-base
```

## Evaluation steps

To perform Retrieval / Tagging tests, go to the .sh files below and edit the paths, and then run them with these bash commands.
```bash
bash evaluation/retrieval/evaluate_retrieval_bible_xlm.sh
```

```bash
bash evaluation/retrieval/evaluate_retrieval_tatoeba_xlm.sh
```

```bash
bash evaluation/tagging/evaluate_ner.sh
```

```bash
bash evaluation/tagging/evaluate_ner_xlmr.sh
```

To calculate avg f1 score from a test_results.txt file
```bash
python evaluation/tagging/calculate_avg_metrics.py \
--file_path evaluation/tagging/pos/random_rob_all_400_checkpoint-0/test_results.txt
```

To calculate metrics for a subset of languages use:
```bash
python evaluation/calculate_avg_subset_lang_metrics.py \
--file_path evaluation/tagging/pos/random_rob_all_400_checkpoint-0/test_results.txt \
--metric f1
```

## Continued Pretraining

For continued pretraining a subset of [Glot500-c](https://github.com/cisnlp/Glot500) corpus is used as training set.
To create the same subset use the script below.
```bash
python continued_pretraining/create_training_dataset.py \
--output_path continued_pretraining/dataset
```

To continued-pretrain the model initialized with HyperOFA go the the .sh files below and adjust the parameters accordingly, then run it with bash command:
```bash
bash train_bash_roberta.sh
```
```bash
bash train_bash_xlm_roberta.sh
```

## Acknowledgements

This repository is built on top of [OFA](https://github.com/cisnlp/ofa) which was also built on top of [transformers](https://github.com/huggingface/transformers), [xtreme](https://github.com/google-research/xtreme), [Glot500](https://github.com/cisnlp/Glot500), [WECHSEL](https://github.com/CPJKU/wechsel) and [FOCUS](https://github.com/konstantinjdobler/focus).
