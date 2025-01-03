import torch
import numpy as np
import yaml

from ofa.utils import WordEmbedding

# Create embedding matrix from the ColexNet embeddings (multilingual_embeddings)
def create_word_embedding_matrix(multilingual_embeddings: WordEmbedding):
    '''
    :param multilingual_embeddings: WordEmbedding object
    :return: embedding matrix with the shape of (len(words), embedding_dim)
    Note: The last row is reserved for PAD token
    '''
    # Get the words
    words = multilingual_embeddings.get_words()
    # Get the word indices
    word_indices = [multilingual_embeddings.get_word_id(word) for word in words]
    # Check if indices start from 0 and end at len(words) - 1
    assert len(word_indices) == len(words), "There are duplicated words in WordEmbedding object"

    # Create the embedding matrix
    embedding_matrix = torch.zeros((len(words), multilingual_embeddings.get_word_vector(word_indices[0]).shape[0]))
    for word_idx in word_indices:
        embedding_matrix[word_idx] = torch.tensor(multilingual_embeddings.get_word_vector(word_idx))

    # Add padding token embedding as the last row
    padding_embedding = torch.zeros(1, embedding_matrix.shape[1])
    embedding_matrix = torch.cat((embedding_matrix, padding_embedding), dim=0)

    return embedding_matrix

def create_input_target_pairs(subword_to_word_mapping, source_matrix):
    """
    Create input-target pairs for the SetFormer model.
    :param subword_to_word_mapping: A dictionary that maps subword idx to word indices.
    :param source_matrix: The source embedding matrix.
    :return: A dictionary containing inputs (lists of word indices) and targets (source vectors).
    """
    
    dataset = {}
    inputs = []
    targets = []

    for subword_idx, word_idxs in subword_to_word_mapping.items():

        inputs.append(word_idxs)

        if source_matrix is not None:
            targets.append(source_matrix[subword_idx])
        else:
            targets.append(np.array([1, 2], dtype=np.float32))  # Dummy target for prediction set
    
    dataset['inputs'] = inputs
    dataset['targets'] = targets
    
    return dataset

def train_val_test_split(source_subword_to_word_mapping, train_ratio, val_ratio, test_ratio, seed=42):
    '''
    Split the source subword_to_word_mapping into train, validation and test sets
    :param source_subword_to_word_mapping: A dictionary that maps subword idx to word indices
    :param train_ratio: The ratio of the train set
    :param val_ratio: The ratio of the validation set
    :param test_ratio: The ratio of the test set
    :param seed: Random seed for reproducibility
    '''
    np.random.seed(seed)
    subword_indices = list(source_subword_to_word_mapping.keys())
    np.random.shuffle(subword_indices)
    num_subwords = len(subword_indices)

    train_size = int(num_subwords * train_ratio)
    val_size = int(num_subwords * val_ratio)
    test_size = int(num_subwords * test_ratio)

    train_mapping_set = {subword_idx: source_subword_to_word_mapping[subword_idx] for subword_idx in subword_indices[:train_size]}
    val_mapping_set = {subword_idx: source_subword_to_word_mapping[subword_idx] for subword_idx in subword_indices[train_size:train_size+val_size]}
    test_mapping_set = {subword_idx: source_subword_to_word_mapping[subword_idx] for subword_idx in subword_indices[train_size+val_size:]}

    return train_mapping_set, val_mapping_set, test_mapping_set
