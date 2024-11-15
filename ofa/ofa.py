from ofa.utils import (
    get_overlapping_tokens,
    get_subword_to_word_mappings,
    perform_factorize,
    create_target_embeddings,
    WordEmbedding
)

from setformer.utils import (
    create_mapping_dataset,
    calculate_target_coord_matrix
)

from setformer.train_setformer import train_setformer

import os
import argparse
import numpy as np
from gensim.models import KeyedVectors
from transformers import AutoModelForMaskedLM, AutoTokenizer


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


def run_ofa(args, multilingual_embeddings, source_tokenizer, target_tokenizer, source_embeddings):

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
        tokenizer=source_tokenizer,
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
    else:
        lower_coordinates = source_embeddings

    # Create the dataset for Setformer training
    train_set, val_set, prediction_set = create_mapping_dataset(source_subword_to_word_mapping, 
                                                        lower_coordinates,
                                                        target_subword_to_word_mapping)

    # Train the setformer model to learn the transformation from Word Vector Space to Subword Vector Space
    setformer_model = train_setformer(setformer_config_yaml='configs/setformer_config.yaml',
                                      multilingual_embeddings=multilingual_embeddings,
                                      train_set=train_set,
                                      val_set=val_set)

    overlapping_token_mapping = get_overlapping_tokens(target_tokenizer, source_tokenizer, fuzzy_search=True)

    # all zero target subword PLM embedding matrix (later for each embedding we will not let be a zero vector)
    target_matrix = np.zeros((len(target_tokenizer), lower_coordinates.shape[1]), dtype=lower_coordinates.dtype)

    # Copy embeddings for overlapping tokens
    overlapping_tokens = {}
    additional_tokens = None

    for overlapping_token, (target_vocab_idx, source_vocab_idx) in overlapping_token_mapping.items():

        overlapping_tokens[overlapping_token] = target_vocab_idx
        target_matrix[target_vocab_idx] = lower_coordinates[source_vocab_idx]

    print(f"Num overlapping tokens: {len(overlapping_tokens)}")
    # the subword tokens that need to be initialized
    additional_tokens = {token: idx for token, idx in target_tokenizer.get_vocab().items()
                            if token not in overlapping_tokens}
    print(f"Num additional tokens: {len(additional_tokens)}")
    assert len(overlapping_tokens) + len(additional_tokens) == len(target_tokenizer)

    final_target_coordinates_matrix = calculate_target_coord_matrix(setformer_model,
                                                                    prediction_set,
                                                                    target_matrix)
    # final_target_matrix, not_found, found = create_target_embeddings(
    #     source_subword_embeddings,
    #     target_subword_embeddings,
    #     source_tokenizer,
    #     target_tokenizer,
    #     source_matrix.copy(),
    #     target_matrix.copy(),
    #     overlapping_tokens,
    #     additional_tokens,
    #     neighbors=args.neighbors,
    #     temperature=args.temperature,
    # )

    if args.do_save:
        # roberta_eng_{keep_dim}
        # xlm_all_{keep_dim}
        model_name = ''
        if args.source_model_name == 'xlm-roberta-base':
            model_name += 'xlm_'
        elif args.source_model_name == 'roberta-base':
            model_name += 'roberta_'
        else:
            raise ValueError("Models other than xlm-r or roberta are not considered!")

        if source_language_set is None:
            model_name += 'all_'
        elif len(source_language_set) == 1 and 'eng' in source_language_set:
            model_name += 'eng_'
        else:
            model_name += 'unk_'

        model_name += f"{args.keep_dim}"

        model_path = args.save_path + model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        np.save(f"{model_path}/target_matrix.npy", final_target_coordinates_matrix)

        if factorize:
            np.save(f"{model_path}/primitive_embeddings.npy", primitive_embeddings)
            np.save(f"{model_path}/source_matrix.npy", lower_coordinates)


def main():
    parser = argparse.ArgumentParser(description='OFA initialization')

    # multilingual embedding related

    parser.add_argument('--emb_dim', type=int, default=200, help='multilingual embedding params')
    parser.add_argument('--num_epochs', type=int, default=10, help='multilingual embedding params')
    parser.add_argument('--number_of_languages', type=int, default=50, help='multilingual embedding params')
    parser.add_argument('--embedding_path', type=str,
                        default='/Users/eno/Documents/my-repos/smarter-ofa/data/',
                        help='multilingual embedding params') # DELETE THE PATH

    # source model related
    parser.add_argument('--source_model_name', type=str, default='xlm-roberta-base', help='source model params')

    # target model related
    parser.add_argument('--target_model_name', type=str, default='cis-lmu/glot500-base', help='target model params')

    # initializing algorithm related
    parser.add_argument('--max_n_word_vectors', type=int, default=None, help='initializing algorithm params')
    parser.add_argument('--neighbors', type=int, default=10, help='initializing algorithm params')
    parser.add_argument('--temperature', type=float, default=0.1, help='initializing algorithm params')
    parser.add_argument('--source_language_set', type=str, default='None', help='initializing algorithm params')
    parser.add_argument('--target_language_set', type=str, default='None', help='initializing algorithm params')

    # factorize related
    parser.add_argument('--factorize', type=bool_flag, default=True, help='factorize params')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorize what is the D' params")

    # save related
    parser.add_argument('--do_save', type=bool_flag, default=True)
    parser.add_argument('--save_path', type=str,
                        default='/Users/eno/Documents/my-repos/smarter-ofa/outputs') # DELETE THE PATH

    args = parser.parse_args()

    # loading multilingual embeddings
    embedding_path = args.embedding_path + \
                     f"colexnet_vectors_minlang_" \
                     f"{args.number_of_languages}_{args.emb_dim}_{args.num_epochs}_updated.wv" # DELETE THE PATH
    loaded_n2v = KeyedVectors.load(embedding_path)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    # loading tokenizers and source-model embeddings
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model_name)  # source tok
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)  # target tok

    source_model = AutoModelForMaskedLM.from_pretrained(args.source_model_name)

    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    assert len(source_tokenizer) == len(source_embeddings)

    print(f"Number of tokens in source tokenizer: {len(source_tokenizer)}")
    print(f"Number of tokens in target tokenizer: {len(target_tokenizer)}")

    run_ofa(args, multilingual_embeddings=multilingual_embeddings,
            source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
            source_embeddings=source_embeddings)


if __name__ == "__main__":
    main()


