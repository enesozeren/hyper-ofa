import torch
from torch.utils.data import DataLoader
from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from setformer.dataset import OFADataset, custom_collate_fn
from setformer.setformer import SetFormer
from setformer.lightning_modules import SetFormerLightning
from setformer.utils import (
    create_word_embedding_matrix, 
    create_input_target_pairs)

def setformer_inference(checkpoint_path, setformer_config_dict: dict, 
                        multilingual_embeddings, mapping_data, 
                        source_matrix, test_or_inference: str, output_path: str):
    '''
    Perform inference on the test set or prediction set
    :param checkpoint_path: The path to the checkpoint
    :param setformer_config_dict: The SetFormer model configuration dictionary
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
                                                   source_matrix=source_matrix,
                                                   max_context_size=setformer_config_dict['model_hps']['max_context_size'])

    # Create the dataset
    dataset = OFADataset(input_target_pairs['inputs'], input_target_pairs['targets'])
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Create collate_fn for the dataloader
    collate_fn = partial(custom_collate_fn, 
                         pad_idx=setformer_config_dict['model_hps']['cls_idx'], 
                         cls_idx=setformer_config_dict['model_hps']['padding_idx'])
    # Prepare dataloaders    
    data_loader = DataLoader(dataset, batch_size=setformer_config_dict['training_hps']['batch_size'], 
                              shuffle=False, collate_fn=collate_fn, num_workers=setformer_config_dict['training_hps']['num_workers'],
                              persistent_workers=True)

    # Convert the word vector embedding to tensor
    word_vector_emb_matrix = create_word_embedding_matrix(multilingual_embeddings)
        
    # Load the model from checkpoint
    setformer = SetFormer(emb_dim=setformer_config_dict['model_hps']['emb_dim'],
                          num_heads=setformer_config_dict['model_hps']['num_heads'],
                          num_layers=setformer_config_dict['model_hps']['num_layers'],
                          dim_feedforward=setformer_config_dict['model_hps']['dim_feedforward'],
                          output_dim=setformer_config_dict['model_hps']['output_dim'],
                          context_size=setformer_config_dict['model_hps']['max_context_size'],
                          dropout=setformer_config_dict['model_hps']['dropout'],
                          word_vector_emb=word_vector_emb_matrix,
                          padding_idx=setformer_config_dict['model_hps']['padding_idx'])
    
    logger = CSVLogger(save_dir=output_path)
    pl_model = SetFormerLightning(setformer, setformer_config_dict)

    if test_or_inference == 'test':
        # Test the model
        trainer = pl.Trainer(accelerator='auto', logger=logger)
        trainer.test(pl_model, ckpt_path=checkpoint_path, dataloaders=data_loader)

    # Save predictions
    pl_model.save_predictions(data_loader, target_subword_idxs, output_path)