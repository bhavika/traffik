import torch
from torch import nn
import pytorch_lightning as pl
from traffik.GraphEnsembleNet import GraphEnsembleNet
from torch.nn import functional as F
from torch_geometric.nn import (
    GCNConv,
    NNConv,
    Set2Set,
    EdgeConv,
    GatedGraphConv,
    GATConv,
    PNAConv,
    SAGEConv,
    SGConv,
    PointConv,
    ChebConv,
)
from torch_geometric.nn import GraphUNet, global_mean_pool, InstanceNorm, LayerNorm
from torch.nn import Sequential, Linear, ReLU, GRU, Tanh, Sigmoid, LeakyReLU, ELU
from torch_geometric.utils import degree
from torch_geometric.data import Data, DataLoader
import os
import numpy as np
import traffik.config as config
from traffik.city_graph_dataset import CityGraphDataset


class LightningGEN(pl.LightningModule):
    def __init__(
        self,
        base_dir: str,
        city: str,
        forward_mins: np.array,
        learning_rate: float,
        overlap: bool,
    ):
        super(LightningGEN, self).__init__()
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.batch_size = 2
        self.pca_static = False
        self.normalize = "Active"
        self.full_val = True

        self.training_ds = CityGraphDataset(
            os.path.join(os.getenv("DATA_DIR"), config.TRAINING_DIR),
            os.path.join(os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR),
            city,
            forward_mins=forward_mins,
            mode=config.TRAINING_DIR,
            overlap=overlap,
            normalize=self.normalize,
            full_val=self.full_val,
            pca_static=self.pca_static,
        )

        self.validation_ds = CityGraphDataset(
            os.path.join(os.getenv("DATA_DIR"), config.VALIDATION_DIR),
            os.path.join(os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR),
            city,
            forward_mins=forward_mins,
            mode=config.VALIDATION_DIR,
            overlap=overlap,
            normalize=self.normalize,
            full_val=self.full_val,
            pca_static=self.pca_static,
        )
        self.net = GraphEnsembleNet(
            self.training_ds.num_node_features,
            self.training_ds[0].y.shape[-1],
            nh=100,
            k=4,
            k_mix=2,
            depth=4,
        )

    def forward(self, data):
        x, edge_idx, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.net(data)
        return x

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        loss = self.loss_fn(out, train_batch.y)
        logs = {"train_loss": loss}
        return {"loss": loss, "logs": logs, "progress_bar": logs}

    def validation_step(self, validation_batch, batch_idx):
        if self.full_val:
            val_graph, y_label_image, zeros_image, day, timestamp = validation_batch
            out = self.forward(val_graph)
            out = self.validation_ds.convert_graph_minibatch_y_to_image(
                out, zeros_image
            )
            out = out.round()
            out = torch.clamp(out, min=0, max=255)
            out = out[:, self.validation_ds.forward_steps - 1, :, :, :]
            y_label_image = y_label_image[
                :, self.validation_ds.forward_steps - 1, :, :, :
            ]
            loss = self.loss_fn(out, y_label_image)
        else:
            out = self.forward(validation_batch)
            out = out.round()
            out = torch.clamp(out, min=0, max=255)
            loss = self.loss_fn(out, validation_batch.y)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def test_step(self, validation_batch, batch_idx):
        if self.full_val:
            val_graph, y_label_image, zeros_image, day, timestamp = validation_batch
            out = self.forward(val_graph)
            out = self.val_data.convert_graph_minibatch_y_to_image(out, zeros_image)
            out = out.round()
            out = torch.clamp(out, min=0, max=255)
            out = out[:, self.val_data.forward_steps - 1, :, :, :]
            y_label_image = y_label_image[:, self.val_data.forward_steps - 1, :, :, :]
            loss = self.loss_fn(out, y_label_image)
            self.val_store.append(
                [
                    day.cpu().numpy()[0].astype(float),
                    timestamp.cpu().numpy()[0].astype(float),
                    loss.cpu().numpy(),
                ]
            )
        else:
            out = self.forward(validation_batch)
            out = out.round()
            out = torch.clamp(out, min=0, max=255)
            loss = self.loss_fn(out, validation_batch.y)
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print(f"Average Training Loss: {avg_loss}")
        tensorboard_logs = {"avg_train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"Average Val Loss: {avg_loss}")
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        print(f"Average Test Loss: {avg_loss}")
        tensorboard_logs = {"avg_test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(
            self.train_data, shuffle=True, batch_size=self.batch_size, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 4000, 1
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "monitor": "val_recall",
                "interval": "step",
                "frequency": 1,
            },
        ]
        return [optimizer], schedulers

    def cyclical_lr(self, stepsize, min_lr=3e-4, max_lr=3e-3):
        import math

        # Scaler: we can adapt this if we do not want the triangular CLR
        scaler = lambda x: 1.0

        # Lambda function to calculate the LR
        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

        # Additional function to see where on the cycle we are
        def relative(it, stepsize):
            cycle = math.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)

        return lr_lambda
