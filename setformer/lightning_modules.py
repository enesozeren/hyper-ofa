import pytorch_lightning as pl
import torch
from setformer.setformer import SetFormer
import matplotlib.pyplot as plt
import os

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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = (1 - self.cosine_similarity(outputs, targets)).mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.model_config_dict['training_hps']['lr'])
    
    def on_train_start(self):
        # Log model parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_dict({'total_params': total_params, 'trainable_params': trainable_params})

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = (1 - self.cosine_similarity(outputs, targets)).mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Avg Cos Sim
        avg_cos_sim = self.cosine_similarity(outputs, targets).mean()
        self.log('avg_cos_sim', avg_cos_sim, on_step=False, on_epoch=True, prog_bar=True)
        return loss, avg_cos_sim


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