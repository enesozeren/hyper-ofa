import yaml
import argparse
import pickle
import numpy as np
import os
from gensim.models import KeyedVectors

from setformer.inference import setformer_inference
from ofa.utils import (
    WordEmbedding
)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_or_inference', type=str, required=True, help='Whether to perform test or inference')
    parser.add_argument('--source_matrix_path', type=str, default=None, help='Path to the source matrix')
    parser.add_argument('--setformer_config_path', type=str, required=True, help='Path to the SetFormer model config')
    parser.add_argument('--test_inference_mapping_data_path', type=str, required=True, help='Path to the inference set')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained SetFormer model checkpoint')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorized what is the D' params")
    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual word vector embeddings') # DELETE THE PATH    
    args = parser.parse_args()

    # Create the output path
    output_path = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint_path)), 
                               f'{args.test_or_inference}_logs')
    os.makedirs(output_path, exist_ok=True)

    # test_or_inference should be either 'test' or 'inference'
    assert args.test_or_inference in ['test', 'inference'], "test_or_inference should be either 'test' or 'inference'"

    # If test, source_matrix_path is required
    if args.test_or_inference == 'test':
        assert args.source_matrix_path is not None, "source_matrix_path is required for test"
        # Read the source matrix
        with open(args.source_matrix_path, 'rb') as f:
            source_matrix = np.load(f)
    else:
        source_matrix = None
    
    # Load the SetFormer model config
    with open(args.setformer_config_path, 'r') as file:
        setformer_config_dict = yaml.load(file, Loader=yaml.FullLoader)
    setformer_config_dict['model_hps']['output_dim'] = args.keep_dim

    # Load the mapping data to perform inference on
    with open(args.test_inference_mapping_data_path, 'rb') as f:
        mapping_data = pickle.load(f)
    
    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    setformer_inference(args.checkpoint_path, setformer_config_dict, 
                        multilingual_embeddings, mapping_data, source_matrix,
                        args.test_or_inference, output_path)
