import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split

## Data Module ##
class OPNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        y,
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
        scale_data=True
    ):
        super().__init__()
        if scale_data:
            X_mean = X.mean(dim=(0, 1), keepdim=True)
            X_std = X.std(dim=(0, 1), unbiased=False, keepdim=True)
            X_std[X_std == 0] = 1  # Prevent division by zero
            X = (X - X_mean) / X_std
            
        self.X = X.to(torch.float32)
        self.y = y.to(torch.long)
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        # Full dataset
        dataset = TensorDataset(self.X, self.y)
        
        # Determine sizes for each dataset
        test_size = int(len(dataset) * self.test_split)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - test_size - val_size

        # Split dataset into train, validation, and test
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        """
        # Calculate weights for each class in the training set
        targets = torch.tensor([self.y[i] for i in self.train_ds.indices])
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[targets]

        # Create a weighted sampler for balanced sampling
        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        """
    # Loading in training data
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            #sampler=self.train_sampler,  # weighted sampler
            num_workers=10,
            persistent_workers=True
        )
    # Loading in validation data
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=10,
            persistent_workers=True
        )

    # Loading in testing data
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=10,
            persistent_workers=True
        )



## Model Module ##
class OPNN(pl.LightningModule):
    def __init__(
        self,
        num_features,
        num_classes,
        lr=0.0003,
        dropout=0.1,
        l1_reg=0
    ):
        super(OPNN, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.l1_reg = l1_reg

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = {
            'test_loss': [],
            'outputs': [],
            'labels': []
        }

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(num_features**2),
            nn.Dropout(dropout),
            nn.Linear(
                num_features**2,
                4096
            ),
            nn.ReLU(),
            nn.LayerNorm(4096),
            nn.Dropout(dropout),
            nn.Linear(
                4096,
                2048
            ),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Dropout(dropout),
            nn.Linear(
                2048,
                1024
            ),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(
                1024,
                256
            ),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                64
            ),
            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):
        x = F.softmax(
            torch.bmm(x.unsqueeze(2), x.unsqueeze(1)),
            dim=1
        )
        x = x.reshape(x.shape[0], -1)
        outputs = self.fc(x)
        return outputs
        
    def l1_penalty(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return l1_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        classification_loss = self.loss_fn(outputs, y)
        
        # Apply the L1 penalty
        l1_penalty = self.l1_penalty()
        total_loss = classification_loss + self.l1_reg * l1_penalty
        
        self.training_step_outputs.append(total_loss)
        self.log(
            'train_loss',
            total_loss,
            prog_bar=True,
            sync_dist=True
        )
        return total_loss

    def on_training_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log(
            'train_loss',
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.training_step_outputs.clear()
        
        return {
            'train_loss': avg_loss,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        self.validation_step_outputs.append(loss)
        self.log(
            'val_loss',
            loss,
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            'val_loss',
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.validation_step_outputs.clear()
        
        return {
            'val_loss': avg_loss,
        }

    def test_step(self, batch, batch_nb):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        self.test_step_outputs['test_loss'].append(loss)
        self.test_step_outputs['outputs'].append(outputs)
        self.test_step_outputs['labels'].append(y)
        return loss
        
    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(self.test_step_outputs['test_loss']).mean()
        all_outputs = torch.cat(self.test_step_outputs['outputs'], dim=0)
        all_labels = torch.cat(self.test_step_outputs['labels'], dim=0)
        
        self.test_results = {
            'test_loss': avg_test_loss.detach().cpu(),
            'y_pred': all_outputs.detach().cpu(),
            'y_true': all_labels.detach().cpu()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }