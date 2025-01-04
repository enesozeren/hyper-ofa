import torch
from torch.utils.data import DataLoader
from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from hypernetwork.dataset import OFADataset, custom_collate_fn
from hypernetwork.lstm import LSTMModel
from hypernetwork.lightning_modules import HypernetworkLightning
from hypernetwork.utils import (
    create_word_embedding_matrix, 
    create_input_target_pairs)


def hypernetwork_inference(checkpoint_path, hypernetwork_config_dict: dict, 
                        multilingual_embeddings, mapping_data, 
                        source_matrix, test_or_inference: str, output_path: str):
    '''
    Perform inference on the test set or prediction set
    :param checkpoint_path: The path to the checkpoint
    :param hypernetwork_config_dict: The hypernetwork model configuration dictionary
    :param multilingual_embeddings: The multilingual word embeddings
    :param mapping_data: The test or prediction set which contains subword to word mappings
    :param source_matrix: The source matrix
    :param test_or_inference: "test" or "inference"
    :param output_path: The output path to save the logs
    :return: Nothing
    '''

    # Check if the test_or_inference is either 'test' or 'inference'
    assert test_or_inference in ['test', 'inference'], "test_or_inference should be either 'test' or 'inference'"

    # Create the input and target pairs for the test or prediction set
    target_subword_idxs = list(mapping_data.keys())

    input_target_pairs = create_input_target_pairs(subword_to_word_mapping=mapping_data,
                                                   source_matrix=source_matrix)

    # Create the dataset
    dataset = OFADataset(input_target_pairs['inputs'], input_target_pairs['targets'],
                         max_context_size=hypernetwork_config_dict['model_hps']['max_context_size'],
                         augment=False)
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Create collate_fn for the dataloader
    collate_fn = partial(custom_collate_fn, 
                         pad_idx=hypernetwork_config_dict['model_hps']['padding_idx'])
    # Prepare dataloaders    
    data_loader = DataLoader(dataset, batch_size=hypernetwork_config_dict['training_hps']['batch_size'], 
                              shuffle=False, collate_fn=collate_fn, num_workers=hypernetwork_config_dict['training_hps']['num_workers'],
                              persistent_workers=True)

    # Convert the word vector embedding to tensor
    word_vector_emb_matrix = create_word_embedding_matrix(multilingual_embeddings)
        
    # Load the model from checkpoint
    lstm = LSTMModel(emb_dim=hypernetwork_config_dict['model_hps']['emb_dim'],
                          hidden_dim=hypernetwork_config_dict['model_hps']['hidden_dim'],
                          num_layers=hypernetwork_config_dict['model_hps']['num_layers'],
                          output_dim=hypernetwork_config_dict['model_hps']['output_dim'],
                          context_size=hypernetwork_config_dict['model_hps']['max_context_size'],
                          dropout=hypernetwork_config_dict['model_hps']['dropout'],
                          word_vector_emb=word_vector_emb_matrix,
                          padding_idx=hypernetwork_config_dict['model_hps']['padding_idx'])    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}") 

    # Load the model from checkpoint using pytorch
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Adjust the state_dict keys
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove the "model." prefix
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        new_state_dict[new_key] = value    
    lstm.load_state_dict(new_state_dict)

    # Create the lightning module
    pl_model = HypernetworkLightning(lstm, hypernetwork_config_dict)
    pl_model.to(device)
    pl_model.eval()

    logger = CSVLogger(save_dir=output_path)

    if test_or_inference == 'test':
        # Test the model
        trainer = pl.Trainer(accelerator='auto', logger=logger)
        trainer.test(pl_model, dataloaders=data_loader)

    # Save the predictions both for inference and test case
    pl_model.save_predictions(data_loader, target_subword_idxs, output_path, device)