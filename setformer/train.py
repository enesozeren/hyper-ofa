import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
import yaml
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from ofa.utils import WordEmbedding
from setformer.dataset import OFADataset, custom_collate_fn
from setformer.setformer import SetFormer
from setformer.utils import create_word_embedding_matrix
from setformer.lightning_modules import SetFormerLightning, LiveLossPlotCallback


def train(model: SetFormer, model_config_dict: dict, 
          train_loader: DataLoader, val_loader: DataLoader, output_dir: str):
    '''
    Train the SetFormer model
    :param model: The SetFormer model
    :param model_config_dict: The configuration dictionary for the SetFormer model
    :param train_loader: The training dataloader
    :param val_loader: The validation dataloader
    :param output_dir: The output directory for the setformer model training logs
    '''
    torch.set_float32_matmul_precision('high')
    
    # Save the model config for future reference
    with open(os.path.join(output_dir, "model_config.yaml"), 'w') as file:
        yaml.dump(model_config_dict, file)

    # Logger
    logger = CSVLogger(save_dir=output_dir)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Live loss plot callback
    live_loss_callback = LiveLossPlotCallback(
        save_dir=os.path.join(output_dir, 'loss_plot'))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=model_config_dict['training_hps']['epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, live_loss_callback],
        log_every_n_steps=250,
        accelerator='auto'  # Automatically uses GPU if available
    )

    # Lightning model
    pl_model = SetFormerLightning(model, model_config_dict)

    # Train
    trainer.fit(pl_model, train_loader, val_loader)


def train_setformer(setformer_config_dict: dict, multilingual_embeddings: WordEmbedding,
                    train_input_target_pairs: dict, val_input_target_pairs: dict, output_dir: str):
    '''
    Prepare the datasets and dataloaders, load the SetFormer model, and call the train function
    :param setformer_config_dict: The configuration dictionary for the SetFormer model
    :param multilingual_embeddings: Multilingual word embeddings
    :param train_input_target_pairs: The training input-target pairs
    :param val_input_target_pairs: The validation input-target pairs
    :param output_dir: The output directory for the setformer model training logs
    :return: Nothing
    '''

    # Pre-fill extra arguments
    collate_fn = partial(custom_collate_fn, 
                         pad_idx=setformer_config_dict['model_hps']['cls_idx'], 
                         cls_idx=setformer_config_dict['model_hps']['padding_idx'])
    
    # Prepare datasets and dataloaders    
    train_set = OFADataset(train_input_target_pairs['inputs'], train_input_target_pairs['targets'],
                            max_context_size=setformer_config_dict['model_hps']['max_context_size'],
                            augment=True, 
                            augmentation_threshold=setformer_config_dict['training_hps']['augmentation_threshold'], 
                            min_percentage=setformer_config_dict['training_hps']['augmentation_min_percentage'], 
                            max_percentage=setformer_config_dict['training_hps']['augmentation_max_percentage'])
    val_set = OFADataset(val_input_target_pairs['inputs'], val_input_target_pairs['targets'],
                            max_context_size=setformer_config_dict['model_hps']['max_context_size'],
                            augment=False)

    train_loader = DataLoader(train_set, batch_size=setformer_config_dict['training_hps']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, num_workers=setformer_config_dict['training_hps']['num_workers'],
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=setformer_config_dict['training_hps']['batch_size'], 
                            shuffle=False, collate_fn=collate_fn, num_workers=setformer_config_dict['training_hps']['num_workers'],
                            persistent_workers=True)

    # Convert the word vector embedding to tensor
    word_vector_emb_matrix = create_word_embedding_matrix(multilingual_embeddings)

    # Load the SetFormer model
    setformer = SetFormer(emb_dim=setformer_config_dict['model_hps']['emb_dim'],
                          num_heads=setformer_config_dict['model_hps']['num_heads'],
                          num_layers=setformer_config_dict['model_hps']['num_layers'],
                          dim_feedforward=setformer_config_dict['model_hps']['dim_feedforward'],
                          output_dim=setformer_config_dict['model_hps']['output_dim'],
                          context_size=setformer_config_dict['model_hps']['max_context_size'],
                          dropout=setformer_config_dict['model_hps']['dropout'],
                          word_vector_emb=word_vector_emb_matrix,
                          padding_idx=setformer_config_dict['model_hps']['padding_idx'])
    
    # Train the model
    train(setformer, setformer_config_dict, train_loader, val_loader, output_dir)