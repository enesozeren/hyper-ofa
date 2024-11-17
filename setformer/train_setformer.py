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
from setformer.dataset import custom_collate_fn
from setformer.setformer import SetFormer
from setformer.utils import create_word_embedding_matrix
from setformer.lightning_modules import SetFormerLightning, LiveLossPlotCallback


def train_process(model: SetFormer, model_config_dict: dict, train_loader, val_loader):

    # Create a sub file that has current timestamp as file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_config_dict['logging']['log_dir'] = f"{model_config_dict['logging']['log_dir']}/{current_time}"
    os.makedirs(model_config_dict['logging']['log_dir'], exist_ok=True)
    
    # Save the model config
    with open(f"{model_config_dict['logging']['log_dir']}/model_config.yaml", 'w') as file:
        yaml.dump(model_config_dict, file)

    # Logger
    logger = CSVLogger(save_dir=model_config_dict['logging']['log_dir'], name=current_time)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{model_config_dict['logging']['log_dir']}/checkpoints",
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Live loss plot callback
    live_loss_callback = LiveLossPlotCallback(
        save_dir=f"{model_config_dict['logging']['log_dir']}/plots")

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


def train_setformer(setformer_config_dict: dict,
                    multilingual_embeddings: WordEmbedding,
                    train_set,
                    val_set):

    # Pre-fill extra arguments
    collate_fn = partial(custom_collate_fn, 
                         pad_idx=setformer_config_dict['model_hps']['cls_idx'], 
                         cls_idx=setformer_config_dict['model_hps']['padding_idx'])
    # Prepare dataloaders    
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
                          word_vector_emb=word_vector_emb_matrix)
    
    # Train the model
    train_process(setformer, setformer_config_dict, train_loader, val_loader)