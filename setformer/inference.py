import torch
from torch.utils.data import DataLoader
from functools import partial
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from setformer.dataset import custom_collate_fn
from setformer.setformer import SetFormer
from setformer.lightning_modules import SetFormerLightning
from setformer.utils import create_word_embedding_matrix

def setformer_inference(checkpoint_path, setformer_config_dict: dict, 
                        multilingual_embeddings, dataset, test_or_inference: str):
    '''
    Perform inference on the test set or prediction set
    :param checkpoint_path: The path to the checkpoint
    :param setformer_config_dict: The SetFormer model configuration dictionary
    :param multilingual_embeddings: The multilingual word embeddings
    :param dataset: The dataset to perform inference on
    :param test_or_inference: "test" or "inference"
    '''

    # Check if the test_or_inference is either 'test' or 'inference'
    assert test_or_inference in ['test', 'inference'], "test_or_inference should be either 'test' or 'inference'"

    # collate_fn
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
    
    logger = CSVLogger(save_dir=setformer_config_dict['logging']['log_dir'], 
                       name=f"setformer_{test_or_inference}_logs")
    pl_model = SetFormerLightning(setformer, setformer_config_dict)

    if test_or_inference == 'test':
        # Test the model
        trainer = pl.Trainer(accelerator='auto', logger=logger)
        trainer.test(pl_model, ckpt_path=checkpoint_path, dataloaders=data_loader)

    # Save predictions
    pl_model.save_predictions(data_loader, 
                              output_path=os.path.join(setformer_config_dict['logging']['log_dir'], f'{test_or_inference}_predictions.npy'))