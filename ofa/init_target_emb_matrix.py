import yaml
import argparse
import pickle
from gensim.models import KeyedVectors

from setformer.inference import test_setformer
from ofa.utils import (
    WordEmbedding
)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_set_path', type=str, required=True, help='Path to the prediction set')
    parser.add_argument('--predictions_path', type=str, required=True, help='Path to the predictions for the prediction set')
    parser.add_argument('--keep_dim', type=int, default=100, help="if factorized what is the D' params")
    args = parser.parse_args()

    # Load the test set
    with open(args.data_set_path, 'rb') as f:
        dataset = pickle.load(f)





















# this function use the initialized source and target subword embeddings to initialize the target PLM embedding matrix
def create_target_embeddings(
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    target_matrix=None,
    overlapping_tokens=None,
    additional_tokens=None,
    neighbors=10,
    temperature=0.1,
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
    :param neighbors: number of neighbors
    :param temperature:
    :return:
    """
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():
            return None

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

        best = sorted(
            [
                (token, similarities[idx])
                for token, idx in zip(best_tokens, best_indices)
            ],
            key=lambda x: -x[1],
        )

        return best

    source_vocab = source_tokenizer.vocab

    # all embeddings are initialized to zero first if no overlapped subword tokens are considered
    if target_matrix is None:
        target_matrix = np.zeros((len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype)
    else:
        # this makes sure that the shape of embeddings match
        assert np.shape(target_matrix) == (len(target_tokenizer), source_matrix.shape[1])

    mean, std = source_matrix.mean(0), source_matrix.std(0)
    random_fallback_matrix = \
        np.random.RandomState(114514).normal(mean, std, (len(target_tokenizer.vocab), source_matrix.shape[1]))

    batch_size = 1024
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
