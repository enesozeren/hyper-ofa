from ofa.utils import WordEmbedding
from setformer.utils import (
    train_val_test_split, 
    create_input_target_pairs)
from setformer.train import train_setformer

import os
import argparse
import numpy as np
import pickle
import yaml
from datetime import datetime
from gensim.models import KeyedVectors


def train_mapping_model(multilingual_embeddings, source_subword_to_word_mapping, 
                        source_matrix, setformer_config_dict, output_dir):
    '''
    Train the setformer model to learn the transformation from Word Vector Space to Subword Vector Space
    :param multilingual_embeddings: Multilingual word embeddings
    :param source_subword_to_word_mapping: A dictionary that maps subword idx to word indices
    :param source_matrix: The source matrix
    :param setformer_config_dict: The setformer config dictionary
    :param output_dir: The output directory for the setformer model training logs
    :return: Nothing
    '''
    
    # Create the input and target pairs for training the setformer model
    train_mapping_set, val_mapping_set, test_mapping_set = train_val_test_split(source_subword_to_word_mapping,
                                                                                train_ratio=0.95, val_ratio=0.025, test_ratio=0.025)
    train_input_target_pairs = create_input_target_pairs(train_mapping_set, source_matrix)
    val_input_target_pairs = create_input_target_pairs(val_mapping_set, source_matrix)
    
    print(f"Train size: {len(train_input_target_pairs['inputs'])}\n", 
          f"Val size: {len(val_input_target_pairs['inputs'])}\n",
          f"Test size: {len(test_mapping_set)}")
    # Save the test mapping set
    with open(os.path.join(output_dir, 'test_mapping_set.pkl'), 'wb') as f:
        pickle.dump(test_mapping_set, f)
    
    # Train the setformer model to learn the transformation from Word Vector Space to Subword Vector Space
    train_setformer(setformer_config_dict=setformer_config_dict,
                    multilingual_embeddings=multilingual_embeddings,
                    train_input_target_pairs=train_input_target_pairs,
                    val_input_target_pairs=val_input_target_pairs,
                    output_dir=output_dir)
    print("Setformer model training completed")


def main():
    parser = argparse.ArgumentParser(description='Training a transformer model without posistional encoding to \
                                     learn the transformation from Word Vector Space to Subword Vector Space')

    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual word vector embeddings') # DELETE THE PATH
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorized what is the D' params")
    parser.add_argument('--mapping_data_dir', type=str, 
                        default='outputs/xlm-roberta-base_to_cis-lmu-glot500-base_dim-100/mapping_data', 
                        help='directory which contains mapping dataset')
    parser.add_argument('--setformer_config_path', type=str, default='setformer/configs/setformer_config.yaml',
                        help='setformer config path')

    args = parser.parse_args()

    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    # Read the source subword to word mapping
    with open(os.path.join(args.mapping_data_dir, 'source_subword_to_word_mapping.pkl'), 'rb') as f:
        source_subword_to_word_mapping = pickle.load(f)
    # Read the source matrix
    with open(os.path.join(args.mapping_data_dir, 'source_matrix.npy'), 'rb') as f:
        source_matrix = np.load(f)
    
    # Create the output dir for the setformer model training logs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(args.mapping_data_dir), 
                              'setformer_training_logs', current_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the setformer config
    with open(args.setformer_config_path, 'r') as file:
        setformer_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    setformer_config_dict['model_hps']['output_dim'] = args.keep_dim

    train_mapping_model(multilingual_embeddings, source_subword_to_word_mapping, 
                        source_matrix, setformer_config_dict, output_dir)


if __name__ == "__main__":
    main()
