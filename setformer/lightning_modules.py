import pytorch_lightning as pl
import torch
import torch.nn as nn
from setformer.setformer import SetFormer
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

class CosineSimilarityMagnitudeLoss(nn.Module):
    def __init__(self, lambda_cos=1.0, lambda_mag=1.0):
        super(CosineSimilarityMagnitudeLoss, self).__init__()
        self.lambda_cos = lambda_cos
        self.lambda_mag = lambda_mag
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, predicted, target):
        # Ensure the tensors are float32 for precision
        predicted = predicted.float()
        target = target.float()

        # Cosine similarity term
        cos_similarity = self.cos_sim(predicted, target)
        cos_loss = 1 - cos_similarity  # (1 - cosine similarity)

        # Element-wise L2 loss (Euclidean distance between predicted and target vectors)
        elementwise_l1_loss = torch.norm(predicted - target, p=1, dim=-1)  # L1 norm between vectors element-wise
        
        # Normalize the elementwise L1 loss by the magnitude of the target vector
        target_magnitude = torch.norm(target, p=1, dim=-1)  # L1 norm (magnitude) of target
        norm_factor = target_magnitude.detach()
        
        normalized_magnitude_loss = elementwise_l1_loss / norm_factor

        # Combine losses
        total_loss = self.lambda_cos * cos_loss + self.lambda_mag * normalized_magnitude_loss
        return total_loss.mean(), cos_loss.mean(), normalized_magnitude_loss.mean()  # Return both individual losses

class SetFormerLightning(pl.LightningModule):
    def __init__(self, model: SetFormer, model_config_dict: dict):
        super().__init__()
        self.model = model
        self.model_config_dict = model_config_dict
        self.criterion = CosineSimilarityMagnitudeLoss(
            lambda_cos=self.model_config_dict['training_hps']['cos_sim_loss_weight'], 
            lambda_mag=self.model_config_dict['training_hps']['magnitude_loss_weight']
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        total_loss, cos_loss, mag_loss = self.criterion(outputs, targets)
        
        # Log the losses separately
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_cos_loss', cos_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mag_loss', mag_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        total_loss, cos_loss, mag_loss = self.criterion(outputs, targets)
        
        # Log the losses separately
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_cos_loss', cos_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mag_loss', mag_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        total_loss, cos_loss, mag_loss = self.criterion(outputs, targets)

        # Log the losses separately
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_cos_loss', cos_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_mag_loss', mag_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log cosine similarity (if needed independently)
        cosine_similarity = torch.nn.functional.cosine_similarity(outputs, targets, dim=1).mean()
        self.log('avg_cos_sim', cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"test_loss": total_loss, "avg_cos_sim": cosine_similarity, "test_cos_loss": cos_loss, "test_mag_loss": mag_loss}
    
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
        self.train_cos_losses = []  # Stores training cosine similarity losses per epoch
        self.val_cos_losses = []  # Stores validation cosine similarity losses per epoch
        self.train_mag_losses = []  # Stores training magnitude losses per epoch
        self.val_mag_losses = []  # Stores validation magnitude losses per epoch
        self.iterations = []  # Stores (epoch, batch) for x-axis, but no batch-specific intervals
        self.save_dir = save_dir  # Directory where plots will be saved
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect losses at the end of each epoch
        train_loss = trainer.callback_metrics.get('train_loss')
        train_cos_loss = trainer.callback_metrics.get('train_cos_loss')
        train_mag_loss = trainer.callback_metrics.get('train_mag_loss')
        
        self.train_losses.append(train_loss.item())
        self.train_cos_losses.append(train_cos_loss.item())
        self.train_mag_losses.append(train_mag_loss.item())
        
        self._save_plot()

    def on_validation_epoch_end(self, trainer, pl_module):
        # If first eval before training starts, append 'E-1' to the iterations list
        if self.val_losses == []:
            self.iterations.append(f'E-BeforeTraining')
        else:
            self.iterations.append(f'E-{trainer.current_epoch}')
           
        val_loss = trainer.callback_metrics.get('val_loss')
        val_cos_loss = trainer.callback_metrics.get('val_cos_loss')
        val_mag_loss = trainer.callback_metrics.get('val_mag_loss')
        
        self.val_losses.append(val_loss.item())
        self.val_cos_losses.append(val_cos_loss.item())
        self.val_mag_losses.append(val_mag_loss.item())

    def _save_plot(self):
        # Create a figure with 3 subplots (one for each loss)
        plt.figure(figsize=(18, 6))

        # Adjust validation data to exclude the first element
        val_iterations = self.iterations[1:]  # Skip the first iteration ('E-BeforeTraining')
        val_losses = self.val_losses[1:] if len(self.val_losses) > 1 else []
        val_cos_losses = self.val_cos_losses[1:] if len(self.val_cos_losses) > 1 else []
        val_mag_losses = self.val_mag_losses[1:] if len(self.val_mag_losses) > 1 else []

        # First subplot for Total Loss (train and validation)
        plt.subplot(1, 3, 1)  # (rows, columns, index)
        if val_losses:
            plt.plot(val_iterations, val_losses, label="Validation Loss", marker='o', color='red')
        if self.train_losses:
            plt.plot(self.iterations[1:], self.train_losses, label="Training Loss", marker='o', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Loss (Training and Validation)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)

        # Second subplot for Cosine Loss (train and validation)
        plt.subplot(1, 3, 2)
        if val_cos_losses:
            plt.plot(val_iterations, val_cos_losses, label="Validation Cosine Loss", marker='x', color='red')
        if self.train_cos_losses:
            plt.plot(self.iterations[1:], self.train_cos_losses, label="Training Cosine Loss", marker='x', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Cosine Loss (Training and Validation)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)

        # Third subplot for Magnitude Loss (train and validation)
        plt.subplot(1, 3, 3)
        if val_mag_losses:
            plt.plot(val_iterations, val_mag_losses, label="Validation Magnitude Loss", marker='s', color='red')
        if self.train_mag_losses:
            plt.plot(self.iterations[1:], self.train_mag_losses, label="Training Magnitude Loss", marker='s', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Magnitude Loss (Training and Validation)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Save the plot to a file, overwriting it each time
        plot_filename = f"{self.save_dir}/loss_plot.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to prevent memory issues
