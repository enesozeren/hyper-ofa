from ofa.utils import (
    WordEmbedding
)

from setformer.train import train_setformer

import os
import argparse
import numpy as np
import pickle
import yaml
from gensim.models import KeyedVectors


def train_mapping_model(args, multilingual_embeddings):

    # Read the train and validation sets
    with open(os.path.join(args.ofa_data_dir, 'train_set.pkl'), 'rb') as f:
        train_set = pickle.load(f)
    
    with open(os.path.join(args.ofa_data_dir, 'val_set.pkl'), 'rb') as f:
        val_set = pickle.load(f)

    # Get the model config
    with open(args.setformer_config_path, 'r') as file:
        setformer_config_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    # Update the output dimension of the model with the keep_dim parameter
    setformer_config_dict['model_hps']['output_dim'] = args.keep_dim

    # Train the setformer model to learn the transformation from Word Vector Space to Subword Vector Space
    train_setformer(setformer_config_dict=setformer_config_dict,
                    multilingual_embeddings=multilingual_embeddings,
                    train_set=train_set,
                    val_set=val_set)
    print("Setformer model training completed")


def main():
    parser = argparse.ArgumentParser(description='OFA initialization')

    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual word vector embeddings') # DELETE THE PATH
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorized what is the D' params")
    parser.add_argument('--ofa_data_dir', type=str, 
                        default='outputs/data_for_xlm-roberta-base_to_cis-lmu/glot500-base', 
                        help='directory which contains ofa dataset')
    parser.add_argument('--setformer_config_path', type=str, default='setformer/configs/setformer_config.yaml',
                        help='setformer config path')

    args = parser.parse_args()

    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    train_mapping_model(args, multilingual_embeddings)


if __name__ == "__main__":
    main()
