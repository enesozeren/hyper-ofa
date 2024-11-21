import yaml
import argparse
import pickle
from gensim.models import KeyedVectors

from setformer.test import test_setformer
from ofa.utils import (
    WordEmbedding
)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--setformer_config_path', type=str, required=True, help='Path to the SetFormer model config')
    parser.add_argument('--test_set_path', type=str, required=True, help='Path to the test set')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained SetFormer model checkpoint')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorized what is the D' params")
    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual word vector embeddings') # DELETE THE PATH    
    args = parser.parse_args()

    # Load the SetFormer model config
    with open(args.setformer_config_path, 'r') as file:
        setformer_config_dict = yaml.load(file, Loader=yaml.FullLoader)
    setformer_config_dict['model_hps']['output_dim'] = args.keep_dim

    # Load the test set
    with open(args.test_set_path, 'rb') as f:
        test_set = pickle.load(f)

    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    test_setformer(args.checkpoint_path, setformer_config_dict, multilingual_embeddings, test_set)
