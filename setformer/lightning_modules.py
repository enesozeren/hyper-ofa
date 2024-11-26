import pytorch_lightning as pl
import torch
from setformer.setformer import SetFormer
import matplotlib.pyplot as plt
import os
import pickle

class SetFormerLightning(pl.LightningModule):
    def __init__(self, model: SetFormer, model_config_dict: dict):
        super().__init__()
        self.model = model
        self.model_config_dict = model_config_dict
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = (1 - self.cosine_similarity(outputs, targets)).mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = (1 - self.cosine_similarity(outputs, targets)).mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.model_config_dict['training_hps']['lr'])
    
    def on_train_start(self):
        # Log model parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_dict({'total_params': total_params, 'trainable_params': trainable_params}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = (1 - self.cosine_similarity(outputs, targets)).mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Avg Cos Sim
        avg_cos_sim = self.cosine_similarity(outputs, targets).mean()
        self.log('avg_cos_sim', avg_cos_sim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss, avg_cos_sim
    
    def save_predictions(self, dataloader, target_subword_idxs: list, output_path):
        '''
        Save the predictions to a dictionary where the key is the target subword index
        :param dataloader: The dataloader
        :param target_subword_idxs: The target subword indices
        :param output_path: The output path to save the predictions
        :return: Nothing
        '''
        
        self.model.eval()
        predictions = {}
        output_list = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                output_list.append(outputs)

        # Concatenate the output_list
        output_tensor = torch.cat(output_list, dim=0)
        # Save the predictions to the dictionary with the target subword index as the key
        for i, target_subword_idx in enumerate(target_subword_idxs):
            predictions[target_subword_idx] = output_tensor[i].cpu().numpy()

        # Save the predictions as pkl
        with open(os.path.join(output_path, 'prediction_dict.pkl'), 'wb') as f:
            pickle.dump(predictions, f)


class LiveLossPlotCallback(pl.Callback):
    def __init__(self, save_dir="plots"):
        self.train_losses = []  # Stores training losses per epoch
        self.val_losses = []  # Stores validation losses per epoch
        self.iterations = []  # Stores (epoch, batch) for x-axis, but no batch-specific intervals
        self.save_dir = save_dir  # Directory where plots will be saved
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect losses at the end of each epoch
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())
        # Call the plot saving method at the end of each epoch
        self._save_plot()        

    def on_validation_epoch_end(self, trainer, pl_module):
        # If first eval before training starts, append 'E-1' to the iterations list
        if self.val_losses == []:
            self.iterations.append(f'E-BeforeTraining')
        else:
            self.iterations.append(f'E-{trainer.current_epoch}')

        val_loss = trainer.callback_metrics.get('val_loss')
        self.val_losses.append(val_loss.item())

    def _save_plot(self):
        # Plot the losses at the end of each epoch
        plt.figure(figsize=(10, 6))

        # Plot training and validation losses
        plt.plot(self.iterations, self.val_losses, label="Validation Loss", marker='x')
        if self.train_losses != []:
            # remove Before Training while plotting training loss
            iters_for_train = self.iterations.copy()
            iters_for_train.pop(0)
            plt.plot(iters_for_train, self.train_losses, label="Training Loss", marker='o')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Losses")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.legend()
        plt.grid(True)

        # Save the plot to a file, overwriting it each time
        plot_filename = f"{self.save_dir}/loss_plot.png"
        plt.tight_layout()  # Prevents clipping of axis labels
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to prevent memory issues