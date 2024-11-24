import yaml
import argparse
import pickle
from gensim.models import KeyedVectors

from setformer.inference import test_setformer
from ofa.utils import (
    WordEmbedding
)

# PLAN

# target_matrix = zero
# overlapping_tokens -> init target matrix as overlapping token embeddings from source matrix
## target_matrix[overlapping_tokens_target_idx] = source_matrix[overlapping_tokens_source_idx]
# additional_tokens -> use setformer model to get embeddings for these tokens
## target_matrix[additional_tokens_target_idx] = setformer_predictions[additional_tokens_target_idx] which is a dict
# tokens which do not match any external word vocab -> random initialization
## target_matrix[not_found_tokens_target_idx] = random_init
# print how many tokens are kept, updated well, updated randomly
# save target_matrix

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model_name', type=str, default='xlm-roberta-base', help='source model params')
    parser.add_argument('--target_model_name', type=str, default='cis-lmu/glot500-base', help='target model params')
    parser.add_argument('--source_matrix_path', type=str, 
                        default='outputs/data_for_xlm-roberta-base_to_cis-lmu-glot500-base/source_matrix.npy',
                        help='source matrix path')
    parser.add_argument('--setformer_predictions_path', type=str, 
                        default='outputs/setformer_logs/inference_predictions.npy', 
                        help='target matrix saving path for target model')
    parser.add_argument('--target_matrix_output_path', type=str, default='outputs/target_matrix', 
                        help='target matrix saving path for target model')
    args = parser.parse_args()



















# this function use the initialized source and target subword embeddings to initialize the target PLM embedding matrix
def create_target_embeddings(
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    target_matrix=None,
    overlapping_tokens=None,
    additional_tokens=None
):
    """
    :param source_subword_embeddings: initialized source subword embeddings
    :param target_subword_embeddings: initialized source subword embeddings
    :param source_tokenizer:
    :param target_tokenizer:
    :param source_matrix: the source-language PLM subword embedding
    :param target_matrix: the initialized subword embedding for target languages
    :param overlapping_tokens: the overlapped tokens in source and target-language tokenizers
    :param additional_tokens: the subword tokens that need to be initialized
    :return:
    """

    source_vocab = source_tokenizer.vocab

    # all embeddings are initialized to zero first if no overlapped subword tokens are considered
    target_matrix = np.zeros((len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype)

    mean, std = source_matrix.mean(0), source_matrix.std(0)

    n_matched = 0

    not_found = {}
    found = {}

    how_many_kept = 0
    how_many_updated = 0
    how_many_randomly_updated = 0

    for i in range(int(math.ceil(len(target_matrix) / batch_size))):
        # use a batch to perform the similarity, otherwise a lot of memory will be consumed
        start, end = (
            i * batch_size,
            min((i + 1) * batch_size, len(target_matrix)),
        )

        similarities = cosine_similarity(target_subword_embeddings[start:end], source_subword_embeddings)

        # here the token_id is actually the index of the target-language PLM embeddings
        for token_id in range(start, end):
            if target_tokenizer.convert_ids_to_tokens(token_id) in overlapping_tokens:
                # we only need to initialize additional_tokens
                found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)
                n_matched += 1
                how_many_kept += 1
                continue

            # for the token not overlapped, the initial embedding should be zero
            assert np.all(target_matrix[token_id] == 0)

            # get the closest neighbors of the subword token
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

            # this corresponds to the case when the subword embedding is not zero
            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)

                found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)

                emb = np.zeros(target_matrix.shape[1])

                for sim_i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab[close_token]] * weights[sim_i]

                target_matrix[token_id] = emb

                n_matched += 1
                how_many_updated += 1
            else:
                # this is a random initialization
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)
                how_many_randomly_updated += 1

    # this is to copy the special tokens
    # we only need to do this if we don't include overlapped tokens
    if additional_tokens is None and overlapping_tokens is None:
        for token in source_tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token = [token]

            for t in token:
                if t in target_tokenizer.vocab and t in additional_tokens:
                    target_matrix[target_tokenizer.vocab[t]] = source_matrix[source_tokenizer.vocab[t]]

    logging.info(
        f"Matching token found for {n_matched} of {len(target_matrix)} tokens."
    )

    print(f"kept: {how_many_kept}")
    print(f"Updated well: {how_many_updated}")
    print(f"Updated randomly: {how_many_randomly_updated}")
    return target_matrix, not_found, found
