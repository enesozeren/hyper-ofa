from hyperofa.utils import (
    get_subword_to_word_mappings,
    perform_factorize,
    WordEmbedding
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


def create_mapping_dataset(args, multilingual_embeddings, source_tokenizer, target_tokenizer, 
                    source_embeddings, output_path: str):

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

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'source_subword_to_word_mapping.pkl'), 'wb') as f:
        pickle.dump(source_subword_to_word_mapping, f)
    with open(os.path.join(output_path, 'target_subword_to_word_mapping.pkl'), 'wb') as f:
        pickle.dump(target_subword_to_word_mapping, f)

    # Save target_not_covered_subwords to initialize them randomly later
    with open(os.path.join(output_path, 'target_not_covered_subwords.pkl'), 'wb') as f:
        pickle.dump(target_not_covered_subwords, f)

    # Save source_matrix as npy
    np.save(os.path.join(output_path, "source_matrix.npy"), source_matrix)
    # Save primitive embeddings as npy if factorized
    if factorize:
        np.save(os.path.join(output_path, 'primitive_embeddings.npy'), primitive_embeddings)


def main():
    parser = argparse.ArgumentParser(
        description='Create OFA Data to learn the transformation from Word Vector Space to Subword Vector Space')

    # multilingual embedding related
    parser.add_argument('--word_vec_embedding_path', type=str,
                        default='colexnet_vectors/colexnet_vectors_minlang_50_200_10_updated.wv',
                        help='multilingual embedding params') # DELETE THE PATH
    # source model related
    parser.add_argument('--source_model_name', type=str, default='xlm-roberta-base', help='source model params')

    # target model related
    parser.add_argument('--target_model_name', type=str, default='cis-lmu/glot500-base', help='target model params')

    # factorize related
    parser.add_argument('--factorize', type=bool_flag, default=True, help='factorize params')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorize what is the D' params")

    # save related
    parser.add_argument('--output_dir', type=str, default='outputs', 
                        help='output directory to save the ofa dataset')

    args = parser.parse_args()

    output_path = os.path.join(
        args.output_dir,
        f'{args.source_model_name.replace("/", "-")}_to_{args.target_model_name.replace("/", "-")}_dim-{args.keep_dim}',
        'mapping_data')

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

    create_mapping_dataset(args, multilingual_embeddings=multilingual_embeddings,
                    source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
                    source_embeddings=source_embeddings, 
                    output_path=output_path)


if __name__ == "__main__":
    main()
