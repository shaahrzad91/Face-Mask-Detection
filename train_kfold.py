import argparse
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
from sklearn.model_selection import KFold

from model import MaskNet


class LitMaskNet(pl.LightningModule):
    def __init__(self, datasets, metrics, batch_size, learning_rate, num_wrokers=4):
        super().__init__()
        self._model = MaskNet()
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataset, self.val_dataset = datasets
        self.metrics = metrics
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_wrokers = num_wrokers

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        return opt

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.criterion(outputs, targets)
        logs = {
            f"{prefix}_{metric_name}": meter(outputs, targets)
            for metric_name, meter in self.metrics.items()
        }
        logs[f"{prefix}_loss"] = loss
        self.log_dict(logs, prog_bar=True, logger=True)

        results = {"loss": loss, "metrics": logs}
        return results

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_wrokers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_wrokers,
        )


def m_wrapper(fn, apply, **kwargs):
    def wrapper(outputs, targets):
        predictions = apply(outputs, dim=1)
        return fn(predictions, targets, **kwargs)

    return wrapper


def get_datasets(dataset_path, n_folds, seed):
    dataset = datasets.ImageFolder(dataset_path)

    # transforms
    image_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.531, 0.478, 0.430], std=[0.251, 0.243, 0.241]),
    ]
    train_transforms = transforms.Compose(image_transforms)
    test_transforms = transforms.Compose(image_transforms[1:])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for idx, (train_index, test_index) in enumerate(kf.split(dataset)):
        train = torch.utils.data.Subset(dataset, train_index)
        test = torch.utils.data.Subset(dataset, test_index)

        train.dataset.transform = train_transforms
        test.dataset.transform = test_transforms
        yield train, test


def cross_validate(exp_name, dataset_path, n_folds, seed):
    for idx, (train_dataset, test_dataset) in enumerate(
        get_datasets(dataset_path, n_folds, seed)
    ):
        fold_exp_name = f"{exp_name}_fold{idx + 1}"
        train_fold(fold_exp_name, (train_dataset, test_dataset))


def train_fold(experiment_name, datasets):
    learning_rate = 1e-3
    batch_size = 128
    num_classes = 3
    metrics = {
        "accuracy": m_wrapper(plF.classification.accuracy, torch.argmax),
        # "weighted_accuracy": m_wrapper(
        #     plF.classification.accuracy, torch.argmax, class_reduction="weighted"
        # ),
        # "average_precision": plF.classification.average_precision,
        "f1_score": partial(plF.f1, num_classes=num_classes),
    }

    num_sanity_val_steps = 10
    max_epochs = 60

    num_gpus = torch.cuda.device_count()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=f"checkpoints/{experiment_name}",
        filename="masknet-{epoch:02d}-{val_accuracy:.3f}",
        save_top_k=5,
        mode="max",
    )
    logger = pl.loggers.TensorBoardLogger("logs", name=experiment_name)

    model = LitMaskNet(
        datasets, metrics=metrics, learning_rate=learning_rate, batch_size=batch_size,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        gpus=num_gpus,
        num_sanity_val_steps=num_sanity_val_steps,
        max_epochs=max_epochs,
        flush_logs_every_n_steps=10,
        progress_bar_refresh_rate=5,
        weights_summary=None,
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KFold cross-validation")

    parser.add_argument(
        "--exp", default="kfold_v2data", type=str, help="experiment name"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="dataset/v2", help="path to dataset"
    )
    parser.add_argument("--n_folds", type=int, default=10, help="number of folds")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # Execute the parse_args() method
    args = parser.parse_args()
    cross_validate(args.exp, args.dataset_path, args.n_folds, args.seed)
