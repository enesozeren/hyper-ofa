import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypernetwork.lstm import LSTMModel
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

class CustomLoss(nn.Module):
    def __init__(self, temperature=0.1, lambd=0.1):
        super(CustomLoss, self).__init__()
        self.lambd = lambd
        self.temperature = temperature

    def forward(self, predictions, targets):

        # * Contrastive Loss *
        # Normalize the vectors for cosine similarity
        normalized_predictions = F.normalize(predictions, p=2, dim=-1)
        normalized_targets = F.normalize(targets, p=2, dim=-1)
        # Cosine similarity (this will act as logits for cross-entropy)
        cosine_sim = torch.mm(normalized_predictions, normalized_targets.t())  # shape (batch_size, batch_size)
        # Apply temperature scaling
        cosine_sim_scaled = cosine_sim / self.temperature
        # Labels for cross-entropy loss: the diagonal elements (same prediction and target) are the positive class
        labels = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
        # Cross-entropy loss: treating cosine similarity as logits
        contrastive_loss = F.cross_entropy(cosine_sim_scaled, labels)
        
        # L1 loss on the original (non-normalized) predictions and targets
        l1_loss = F.l1_loss(predictions, targets, reduction='mean') / torch.mean(abs(targets))
        
        # Total loss: weighted combination of contrastive and L1 losses
        loss = self.lambd * contrastive_loss + (1-self.lambd) * l1_loss
        
        return loss, contrastive_loss, l1_loss
    
class HypernetworkLightning(pl.LightningModule):
    def __init__(self, model: LSTMModel, model_config_dict: dict):
        super().__init__()
        self.model = model
        self.model_config_dict = model_config_dict
        self.criterion = CustomLoss(
            temperature=model_config_dict["training_hps"]["contrastive_temp"],
            lambd=model_config_dict["training_hps"]["loss_lambd"]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        train_loss, contrastive_loss, l1_loss = self.criterion(outputs, targets)
        
        # Log the losses separately
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_contrastive_loss', contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        val_loss, contrastive_loss, l1_loss = self.criterion(outputs, targets)
        
        cosine_similarity = F.cosine_similarity(outputs, targets, dim=1).mean()

        # Log the losses separately
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_contrastive_loss', contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log('val_avg_cos_sim', cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.model_config_dict['training_hps']['lr'],
                                     weight_decay=self.model_config_dict['training_hps']['weight_decay'])
        
        # Define the StepLR scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.model_config_dict['training_hps']['lr_sched_step_size'],
                                                        gamma=self.model_config_dict['training_hps']['lr_sched_gamma']),
            'interval': 'epoch',  # Apply every epoch
            'frequency': 1,       # Apply once every epoch
        }
        return [optimizer], [scheduler]

    def on_train_start(self):
        # Log model parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_dict({'total_params': total_params, 'trainable_params': trainable_params}, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate all three losses
        test_loss, test_contrastive_loss, test_l1_loss = self.criterion(outputs, targets)

        # Log the losses separately
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_contrastive_loss', test_contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_l1_loss', test_l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log cosine similarity (if needed independently)
        cosine_similarity = torch.nn.functional.cosine_similarity(outputs, targets, dim=1).mean()
        self.log('avg_cos_sim', cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate average L1 distance element-wise
        test_avg_l1_dist_elementwise = F.l1_loss(outputs, targets, reduction='mean')
        self.log('test_avg_l1_dist_elementwise', test_avg_l1_dist_elementwise, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {
            "test_loss": test_loss, 
            "test_avg_cos_sim": cosine_similarity,
            "test_avg_l1_dist_elementwise": test_avg_l1_dist_elementwise
            }
    
    def save_predictions(self, dataloader, target_subword_idxs: list, output_path, device):
        '''
        Save the predictions to a dictionary where the key is the target subword index
        :param dataloader: The dataloader
        :param target_subword_idxs: The target subword indices
        :param output_path: The output path to save the predictions
        :param device: The device to run the model
        :return: Nothing
        '''
        self.model.to(device)
        self.model.eval()
        predictions = {}
        output_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(device)

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
        self.train_losses = []  # Stores training total losses per epoch
        self.val_losses = []  # Stores validation total losses per epoch
        self.val_cos_sim = []  # Stores validation cosine similarities per epoch
        self.iterations = []  # Stores (epoch, batch) for x-axis, but no batch-specific intervals
        self.save_dir = save_dir  # Directory where plots will be saved
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect losses at the end of each epoch
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())
        self._save_plot()

    def on_validation_epoch_end(self, trainer, pl_module):
        # If first eval before training starts, append 'E-1' to the iterations list
        if self.val_losses == []:
            self.iterations.append(f'E-BeforeTraining')
        else:
            self.iterations.append(f'E-{trainer.current_epoch}')
        
        val_loss = trainer.callback_metrics.get('val_loss')
        self.val_losses.append(val_loss.item())       
        val_cos_sim = trainer.callback_metrics.get('val_avg_cos_sim')
        self.val_cos_sim.append(val_cos_sim.item())

    def _save_plot(self):
        # Create a figure with 2 subplots
        plt.figure(figsize=(24, 12))
        
        iterations = self.iterations[1:]

        # Dynamically determine x-axis tick spacing
        total_epochs = len(iterations)
        max_ticks = 10  # Maximum number of ticks on the x-axis
        tick_spacing = max(1, total_epochs // max_ticks)  # Calculate tick spacing
        x_ticks = [i for i in range(0, total_epochs, tick_spacing)]  # Select tick positions
        x_tick_labels = [iterations[i] for i in x_ticks]  # Get corresponding labels

        # First subplot for Loss (train and validation)
        plt.subplot(1, 2, 1)  # (rows, columns, index)
        plt.plot(iterations, self.val_losses[1:], label="Validation Loss", marker='o', color='red')
        if self.train_losses:
            plt.plot(iterations, self.train_losses, label="Training Loss", marker='o', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Loss (Training and Validation)")
        plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45, ha="right", fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.grid(True)

        # Second subplot
        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.val_cos_sim[1:], label="Validation Cosine Similarity", marker='d', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity (Validation)")
        plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45, ha="right", fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.grid(True)

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Save the plot to a file, overwriting it each time
        plot_filename = f"{self.save_dir}/loss_plot.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to prevent memory issues
