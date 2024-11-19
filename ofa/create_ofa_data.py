from ofa.utils import (
    get_subword_to_word_mappings,
    perform_factorize,
    WordEmbedding
)

from setformer.utils import (
    create_mapping_dataset,
)

import os
import argparse
import numpy as np
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


def create_ofa_data(args, multilingual_embeddings, source_tokenizer, target_tokenizer, source_embeddings):

    source_language_set = eval(args.source_language_set)
    target_language_set = eval(args.target_language_set)

    print(f"Source language set: {source_language_set}")
    print(f"Target language set: {target_language_set}")

    # Get the source subword to word mapping
    source_subword_to_word_mapping, source_not_covered_subwords = get_subword_to_word_mappings(
        tokenizer=source_tokenizer,
        model=multilingual_embeddings
    )

    # Get the target subword to word mapping
    target_subword_to_word_mapping, target_not_covered_subwords = get_subword_to_word_mappings(
        tokenizer=target_tokenizer,
        model=multilingual_embeddings
    )

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

    # Create the dataset for Setformer training
    train_set, val_set, test_set_source_token_ids, prediction_set = create_mapping_dataset(source_subword_to_word_mapping, 
                                                                                            source_matrix,
                                                                                            target_subword_to_word_mapping,
                                                                                            args.setformer_config_path)
    
    # Create the output directory if it does not exist
    if not os.path.exists(args.ofa_data_dir):
        os.makedirs(args.ofa_data_dir, exist_ok=True)
        
    # Save the dataset to the output directory for learning the transformation from word vector space to subword vector space
    with open(os.path.join(args.ofa_data_dir, 'train_set.pkl'), 'wb') as f:
        pickle.dump(train_set, f)
    with open(os.path.join(args.ofa_data_dir, 'val_set.pkl'), 'wb') as f:
        pickle.dump(val_set, f)
    with open(os.path.join(args.ofa_data_dir, 'test_set_source_token_ids.pkl'), 'wb') as f:
        pickle.dump(test_set_source_token_ids, f)
    with open(os.path.join(args.ofa_data_dir, 'prediction_set.pkl'), 'wb') as f:
        pickle.dump(prediction_set, f)

    # Save target_not_covered_subwords to initialize them randomly later
    with open(os.path.join(args.ofa_data_dir, 'target_not_covered_subwords.pkl'), 'wb') as f:
        pickle.dump(target_not_covered_subwords, f)

    # Save source_matrix as npy
    np.save(os.path.join(args.ofa_data_dir, "source_matrix.npy"), source_matrix)

    # Save primitive embeddings as npy if factorized
    if factorize:
        np.save(os.path.join(args.ofa_data_dir, 'primitive_embeddings.npy'), primitive_embeddings)


def main():
    parser = argparse.ArgumentParser(
        description='Create OFA Data to learn the transformation from Word Vector Space to Subword Vector Space')

    # multilingual embedding related
    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual embedding params') # DELETE THE PATH
    parser.add_argument('--source_language_set', type=str, default='None', help='initializing algorithm params')
    parser.add_argument('--target_language_set', type=str, default='None', help='initializing algorithm params')

    # source model related
    parser.add_argument('--source_model_name', type=str, default='xlm-roberta-base', help='source model params')

    # target model related
    parser.add_argument('--target_model_name', type=str, default='cis-lmu/glot500-base', help='target model params')

    # factorize related
    parser.add_argument('--factorize', type=bool_flag, default=True, help='factorize params')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorize what is the D' params")

    # save related
    parser.add_argument('--ofa_data_dir', type=str, default='outputs/ofa_data', 
                        help='output directory to save the ofa dataset')

    # setformer model hp related
    parser.add_argument('--setformer_config_path', type=str, default='setformer/configs/setformer_config.yaml',
                        help='setformer model hyperparameters config path')

    args = parser.parse_args()

    loaded_n2v = KeyedVectors.load(args.word_vec_embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    # loading tokenizers and source-model embeddings
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_name)  # source tok
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)  # target tok

    source_model = AutoModelForMaskedLM.from_pretrained(args.source_model_name)

    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    assert len(source_tokenizer) == len(source_embeddings)

    print(f"Number of tokens in source tokenizer: {len(source_tokenizer)}")
    print(f"Number of tokens in target tokenizer: {len(target_tokenizer)}")

    create_ofa_data(args, multilingual_embeddings=multilingual_embeddings,
            source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
            source_embeddings=source_embeddings)


if __name__ == "__main__":
    main()
