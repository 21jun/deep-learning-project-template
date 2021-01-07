from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)
        self.example_input_array = torch.rand(10, 28 * 28)
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_dim, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_acc_step", acc)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("epoch_acc", self.train_acc.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)
        self.log('valid_loss', loss)

    def validation_epoch_end(self, outputs):
        self.log("epoch_val_acc", self.val_acc.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_acc(y_hat, y)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser


class LitMNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        MNIST(self.data_dir, train=False, download=True)
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default='', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    mnist = LitMNISTDataModule(args.data_dir, args.batch_size)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, mnist)

    # ------------
    # testing
    # ------------
    trainer.test(datamodule=mnist)


if __name__ == '__main__':
    cli_main()
