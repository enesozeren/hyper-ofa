import torch
from torch.utils.data import Dataset
import numpy as np

class OFADataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def custom_collate_fn(batch, cls_idx, pad_idx):
    '''
    Collate function for the dataloader
    Add CLS token to the beginning of the input
    Add PAD token to make the input size equal across the batch
    The targets stay the same
    '''
    inputs, targets = zip(*batch)
    batch_size = len(targets)
    
    # Add CLS token to the beginning of the input
    padded_inputs = torch.nn.utils.rnn.pad_sequence([torch.cat([torch.tensor([cls_idx]), torch.tensor(i)]) for i in inputs], 
                                                    batch_first=True, padding_value=pad_idx)
    targets = torch.tensor(np.array(targets), dtype=torch.float32).view(batch_size, targets[0].shape[0])

    return padded_inputs, targets