import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np
import random

class OFADataset(Dataset):
    def __init__(self, inputs, targets, max_context_size, augment=False, 
                    augmentation_threshold=10, min_percentage=0.5, max_percentage=1.0, keep_all_prob=0.5):
        """
        Args:
            inputs (list of list): List of input vectors (each input is a list of vectors).
            targets (list): List of corresponding targets.
            augment (bool): Whether to apply augmentation (used for training dataset).
            augmentation_threshold (int): Minimum number of vectors required to apply augmentation.
            min_percentage (float): Minimum percentage of vectors to keep during augmentation.
            max_percentage (float): Maximum percentage of vectors to keep during augmentation.
            keep_all_prob (float): Probability of keeping the all vectors
        """            
        self.inputs = inputs
        self.targets = targets
        self.max_context_size = max_context_size
        self.augment = augment
        self.augmentation_threshold = augmentation_threshold
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage
        self.keep_all_prob = keep_all_prob

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        target = self.targets[idx]
        
        # Only augment if in training mode and len(inputs) is above the augmentation_threshold
        if self.augment and random.random() < self.keep_all_prob and len(inputs) > self.augmentation_threshold:
            percentage = random.uniform(self.min_percentage, self.max_percentage)
            num_to_keep = max(1, int(len(inputs) * percentage))  # Keep at least one vector
            inputs = random.sample(inputs, num_to_keep)
        
        # Truncate inputs to the context size
        inputs = inputs[:self.max_context_size]

        return inputs, target

class CustomGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset (OFADataset): The dataset from which to sample.
            batch_size (int): The batch size for sampling.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Create pairs of (index, sequence length) for each input
        self.seq_length_pairs = [(idx, len(seq)) for idx, seq in enumerate(dataset.inputs)]

    def __iter__(self):
        # Shuffle the sequence length pairs to introduce randomness
        random.shuffle(self.seq_length_pairs)

        # Group sequences by their lengths (group size is 100 times the batch size)
        group_size = self.batch_size * 40
        groups = []

        for index in range(0, len(self.seq_length_pairs), group_size):
            group = self.seq_length_pairs[index:index + group_size]
            # Sort the group by sequence length
            sorted_group = sorted(group, key=lambda x: x[1])
            # Add the indices of sorted sequences to the group list
            groups.extend([idx for idx, _ in sorted_group])

        # Yield the indices for each batch
        return iter(groups)

    def __len__(self):
        return len(self.dataset)

def custom_collate_fn(batch, pad_idx):
    '''
    Collate function for the dataloader
    Add PAD token to make the input size equal across the batch
    The targets stay the same
    '''
    inputs, targets = zip(*batch)
    batch_size = len(targets)
    
    # Pad the sequences
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i) for i in inputs], 
        batch_first=True, padding_value=pad_idx
        )
    
    targets = torch.tensor(np.array(targets), dtype=torch.float32).view(batch_size, targets[0].shape[0])

    return padded_inputs, targets