import torch
from torch.utils.data import DataLoader
import yaml
from functools import partial
import argparse
import pickle

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from setformer.dataset import custom_collate_fn
from setformer.setformer import SetFormer
from setformer.lightning_modules import SetFormerLightning
from setformer.utils import create_word_embedding_matrix

def test_setformer(checkpoint_path, setformer_config_dict: dict, multilingual_embeddings, test_set):

    # collate_fn
    collate_fn = partial(custom_collate_fn, 
                         pad_idx=setformer_config_dict['model_hps']['cls_idx'], 
                         cls_idx=setformer_config_dict['model_hps']['padding_idx'])
    # Prepare dataloaders    
    test_loader = DataLoader(test_set, batch_size=setformer_config_dict['training_hps']['batch_size'], 
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
    
    # Test the model
    logger = CSVLogger(save_dir=setformer_config_dict['logging']['log_dir'], name='test_logs')
    pl_model = SetFormerLightning(setformer, setformer_config_dict)
    trainer = pl.Trainer(accelerator='auto', logger=logger)
    trainer.test(pl_model, ckpt_path=checkpoint_path, dataloaders=test_loader)