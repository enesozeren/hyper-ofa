import argparse
import pickle
import os
from transformers import AutoTokenizer
import numpy as np

from hyperofa.utils import (
    get_overlapping_tokens
)


def create_random_target_embeddings(source_tokenizer, target_tokenizer, source_matrix):
    """
    Create the target-language embedding matrix
    :param source_tokenizer:
    :param target_tokenizer:
    :param source_matrix: the source-language PLM subword embedding
    :return: the target-language embedding matrix
    """

    # all embeddings are initialized to zero first
    target_matrix = np.zeros((len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype)
    
    # Find the overlapping tokens between source and target tokenizers
    overlapping_token_mapping = get_overlapping_tokens(target_tokenizer, source_tokenizer, fuzzy_search=True)
    print(f"Overlapping tokens count: {len(overlapping_token_mapping)}")

    # Initialize the overlapping tokens in the target matrix from the source matrix
    for target_idx, source_idx in overlapping_token_mapping.values():
        target_matrix[target_idx] = source_matrix[source_idx]
    
    # Initialize the remaining tokens in the target matrix with random embeddings
    random_init_token_counter = 0
    mean, std = source_matrix.mean(0), source_matrix.std(0)
    random_state = np.random.RandomState(0)
    for target_idx, embedding in enumerate(target_matrix):
        if np.all(embedding == 0):
            target_matrix[target_idx] = random_state.normal(loc=mean, scale=std, size=embedding.shape[0])
            random_init_token_counter += 1
    print(f"Randomly initialized tokens count: {random_init_token_counter}")

    return target_matrix


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Create target-language random init embedding matrix')
    parser.add_argument('--source_model_name', type=str, required=True, help='source model params')
    parser.add_argument('--target_model_name', type=str, default='cis-lmu/glot500-base', help='target model params')
    parser.add_argument('--source_matrix_path', type=str, required=True, help='source matrix path')
    args = parser.parse_args()

    # Load source matrix
    source_matrix = np.load(args.source_matrix_path)
    # Load primitive_embeddings
    primitive_embeddings = np.load(args.source_matrix_path.replace('source_matrix.npy', 'primitive_embeddings.npy'))

    # loading tokenizers and source-model embeddings
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_name)  # source tok
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)  # target tok

    target_matrix = create_random_target_embeddings(source_tokenizer, target_tokenizer, source_matrix)

    output_dir = os.path.join('outputs', f'random_{args.source_model_name[:3]}_all_{target_matrix.shape[1]}')
    
    os.makedirs(output_dir, exist_ok=True)
    # Save matrices
    np.save(os.path.join(output_dir, 'target_matrix.npy'), target_matrix)
    np.save(os.path.join(output_dir, 'source_matrix.npy'), source_matrix)
    np.save(os.path.join(output_dir, 'primitive_embeddings.npy'), primitive_embeddings)

