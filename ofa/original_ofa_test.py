from ofa.utils import (
    get_subword_embeddings_in_word_embedding_space,
    perform_factorize,
    get_overlapping_tokens,
    create_target_embeddings,
    calculate_cos_sim_of_embeddings,
    WordEmbedding
)

import numpy as np
import os
import argparse
from gensim.models import KeyedVectors
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pickle

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def run_ofa(args, multilingual_embeddings, source_tokenizer, source_embeddings):

    print("Constructing the source-language subword embeddings ...")
    # source_subword_embeddings, source_subword_sources, not_covered_source_subwords = \
    #     get_subword_embeddings_in_word_embedding_space(
    #         source_tokenizer,
    #         multilingual_embeddings,
    #         return_not_covered_set=True
    #     )
    
    # Create the output path dir
    # os.makedirs(args.output_path, exist_ok=True)
    # # Save source subword embeddings (TO BE DELETED)
    # with open(os.path.join(args.output_path, 'source_subword_embeddings.pkl'), 'wb') as f:
    #     pickle.dump(source_subword_embeddings, f)

    # Read the source_subword_embeddings
    with open(os.path.join(args.output_path, 'source_subword_embeddings.pkl'), 'rb') as f:
        source_subword_embeddings = pickle.load(f)

    # source_subword_sources stores the subwords from the source language
    # print(f"Coverage: {len(source_subword_sources) / (len(source_subword_sources) + len(not_covered_source_subwords))}")

    # Read the test source token ids
    with open(args.test_set_source_token_ids_path, 'rb') as f:
        test_source_token_ids = pickle.load(f)

    # Get the subword embeddings for test from source_subword_embeddings 
    target_subword_embeddings = source_subword_embeddings[test_source_token_ids]

    # Remove the target subword embeddings from the source subword embeddings
    source_subword_embeddings = np.delete(source_subword_embeddings, test_source_token_ids, axis=0)

    if args.keep_dim == source_embeddings.shape[1]:
        factorize = False
    elif args.keep_dim < source_embeddings.shape[1]:
        factorize = True
    else:
        raise ValueError("The keep_dim must be smaller than the original embedding dim")
    
    print(f"Keep dim is {args.keep_dim} and factorize is {str(factorize)}")
    # factorize the source-language PLM subword embeddings
    if factorize:
        primitive_embeddings, lower_coordinates = perform_factorize(source_embeddings, keep_dim=args.keep_dim)
        source_matrix = lower_coordinates
    else:
        source_matrix = source_embeddings

    # all zero target subword PLM embedding matrix (later for each embedding we will not let be a zero vector)
    target_matrix = np.zeros((len(test_source_token_ids), source_matrix.shape[1]), dtype=source_matrix.dtype)

    final_target_matrix = create_target_embeddings(
        source_subword_embeddings,
        target_subword_embeddings,
        source_tokenizer,
        test_source_token_ids,
        source_matrix.copy(),
        target_matrix.copy(),
        overlapping_tokens=None,
        additional_tokens=None,
        neighbors=args.neighbors,
        temperature=args.temperature,
    )

    # Calculate the cosine similarity on test set
    cos_similarities_list = calculate_cos_sim_of_embeddings(source_matrix[test_source_token_ids], 
                                                         final_target_matrix)
    avg_cos_similarity = np.mean(cos_similarities_list)
    print(f"Average cosine similarity on the test set: {avg_cos_similarity}")
    # Save the cos_similarities_list to the output path
    with open(os.path.join(args.output_path, 'cos_similarities_list.pkl'), 'wb') as f:
        pickle.dump(cos_similarities_list, f)
    # Save the numb of tokens in the test set and avg_cos_similarity
    with open(os.path.join(args.output_path, 'test_set_info.txt'), 'w') as f:
        f.write(f"Number of tokens in the test set: {len(test_source_token_ids)}\n")
        f.write(f"Average cosine similarity on the test set: {avg_cos_similarity}\n")
    
    print("Cosine similarities are saved to the output path")
    print("Test finished!")


def main():
    parser = argparse.ArgumentParser(description='OFA initialization')

    # multilingual embedding related
    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual embedding params') # DELETE THE PATH

    # source model related
    parser.add_argument('--source_model_name', type=str, default='xlm-roberta-base', help='source model params')
    parser.add_argument('--test_set_source_token_ids_path', type=str, 
                        default='outputs/ofa_data/test_set_source_token_ids.pkl', 
                        help='test set source token ids path')
    # initializing algorithm related
    parser.add_argument('--max_n_word_vectors', type=int, default=None, help='initializing algorithm params')
    parser.add_argument('--neighbors', type=int, default=10, help='initializing algorithm params')
    parser.add_argument('--temperature', type=float, default=0.1, help='initializing algorithm params')

    # factorize related
    parser.add_argument('--factorize', type=bool_flag, default=True, help='factorize params')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorize what is the D' params")

    # save related
    parser.add_argument('--output_path', type=str,
                        default='outputs/original_ofa_test', 
                        help='output directory to save the original ofa test results')

    args = parser.parse_args()

    # loading multilingual embeddings
    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    # loading tokenizers and source-model embeddings
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_name)  # source tok

    source_model = AutoModelForMaskedLM.from_pretrained(args.source_model_name)

    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    assert len(source_tokenizer) == len(source_embeddings)

    print(f"Number of tokens in source tokenizer: {len(source_tokenizer)}")

    run_ofa(args, multilingual_embeddings=multilingual_embeddings,
            source_tokenizer=source_tokenizer,
            source_embeddings=source_embeddings)


if __name__ == "__main__":
    main()

