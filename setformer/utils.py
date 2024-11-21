import torch
import numpy as np
import yaml

from ofa.utils import WordEmbedding
from setformer.dataset import OFADataset


# Create embedding matrix from the ColexNet embeddings (multilingual_embeddings)
def create_word_embedding_matrix(multilingual_embeddings: WordEmbedding):
    '''
    :param multilingual_embeddings: WordEmbedding object
    :return: embedding matrix with the shape of (len(words), embedding_dim)
    Note: The last -1 row is reserved for PAD token and the last row is reserved for the CLS token
    '''
    # Get the words
    words = multilingual_embeddings.get_words()
    # Get the word indices
    word_indices = {word: multilingual_embeddings.get_word_id(word) for word in words}
    # Check if indices start from 0 and end at len(words) - 1
    assert len(word_indices) == len(words), "There are duplicated words in WordEmbedding object"
    assert min(word_indices.values()) == 0, "Indices do not start from 0 in WordEmbedding object"
    assert max(word_indices.values()) == len(words) - 1, "Indices do not end at len(words) - 1 in WordEmbedding object"
    
    # Get the word vectors
    word_vectors_np = np.array([multilingual_embeddings.get_word_vector(word) for word in words])
    word_vectors = torch.tensor(word_vectors_np)
    
    # Create the embedding matrix
    embedding_matrix = torch.zeros((len(words), word_vectors.shape[1]))
    for word, word_id in word_indices.items():
        embedding_matrix[word_id] = word_vectors[word_id]
    
    # Add padding token embedding as the second last row and the CLS token embedding as the last row
    padding_embedding = torch.zeros(1, word_vectors.shape[1])
    cls_embedding = torch.zeros(1, word_vectors.shape[1])
    embedding_matrix = torch.cat((embedding_matrix, padding_embedding, cls_embedding), dim=0)

    return embedding_matrix

def get_test_source_token_ids(subword_to_word_mapping, test_set_size: int):
    '''
    Get the test set from the source subword_to_word_mapping
    '''
    # randomly select test_set_size number of subword indices from the subword_to_word_mapping keys
    test_subword_indices = np.random.choice(list(subword_to_word_mapping.keys()), test_set_size, replace=False)
    print(f"Test set size: {test_set_size}")
    return test_subword_indices

def remove_test_set_from_source(subword_to_word_mapping, test_set_source_token_ids):
    '''
    Remove the test set from the source subword_to_word_mapping
    '''
    for subword_idx in test_set_source_token_ids:
        del subword_to_word_mapping[subword_idx]

    return subword_to_word_mapping

# The dataset size can be increased by generating shuffled word indices which has a larger size than the context size
def create_input_target_pairs(subword_to_word_mapping, source_matrix, max_context_size: int):
    '''
    Create input-target pairs for the SetFormer model
    :param subword_to_word_mapping: A dictionary that maps subword idx to word indices
    :param source_matrix: The source embedding matrix
    '''
    
    dataset = {}
    inputs = []
    targets = []
    for subword_idx, word_idxs in subword_to_word_mapping.items():
        # Shuffle the word indices 
        np.random.shuffle(word_idxs)
        # Truncate inputs to the context size
        word_idxs = word_idxs[:max_context_size-1] # -1 for the CLS token to be added in collate_fn
        inputs.append(word_idxs)

        if source_matrix is not None:
            targets.append(source_matrix[subword_idx])
        else:
            targets.append(None)

    dataset['inputs'] = inputs
    dataset['targets'] = targets
 
    return dataset

def split_train_val_set(dataset, val_ratio=0.1, seed=42):
    '''
    Split the dataset into train and validation sets after random shuffling
    '''
    np.random.seed(seed)
    num_samples = len(dataset['inputs'])
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    val_size = int(num_samples * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_set = {'inputs': [dataset['inputs'][i] for i in train_indices],
                 'targets': [dataset['targets'][i] for i in train_indices]}
    print(f"Train set size: {len(train_set['inputs'])}")
    val_set = {'inputs': [dataset['inputs'][i] for i in val_indices],
               'targets': [dataset['targets'][i] for i in val_indices]}
    print(f"Validation set size: {len(val_set['inputs'])}")

    return train_set, val_set

def create_mapping_dataset(source_subword_to_word_mapping, source_matrix,
                           target_subword_to_word_mapping, setformer_config_path):

    # Get the model config
    with open(setformer_config_path, 'r') as file:
        setformer_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # Split the test set to compare learning based OFA and original OFA later
    test_set_source_token_ids = get_test_source_token_ids(source_subword_to_word_mapping, test_set_size=3_000)
    test_subword_to_word_mapping = {subword_idx: source_subword_to_word_mapping[subword_idx] for subword_idx in test_set_source_token_ids}
    test_set = create_input_target_pairs(subword_to_word_mapping=test_subword_to_word_mapping,
                                         source_matrix=source_matrix,
                                         max_context_size=setformer_config_dict['model_hps']['max_context_size'])
    test_set = OFADataset(test_set['inputs'], test_set['targets'])

    # Remove the test set from the source_subword_to_word_mapping
    source_subword_to_word_mapping = remove_test_set_from_source(source_subword_to_word_mapping, test_set_source_token_ids)

    train_set = create_input_target_pairs(subword_to_word_mapping=source_subword_to_word_mapping, 
                                          source_matrix=source_matrix, 
                                          max_context_size=setformer_config_dict['model_hps']['max_context_size'])
    train_set, val_set = split_train_val_set(train_set, val_ratio=0.05)
    
    train_set = OFADataset(train_set['inputs'], train_set['targets'])
    val_set = OFADataset(val_set['inputs'], val_set['targets'])

    prediction_set = create_input_target_pairs(subword_to_word_mapping=target_subword_to_word_mapping, 
                                                source_matrix=None,
                                                max_context_size=setformer_config_dict['model_hps']['max_context_size'])

    return train_set, val_set, test_set, test_set_source_token_ids, prediction_set

def calculate_target_coord_matrix(setformer_model, prediction_set, target_matrix):
    pass