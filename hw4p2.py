import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from phonetics import VOCAB
from las import LAS

training_x_dir = os.path.join("train-clean-100", "mfcc")
training_y_dir = os.path.join("train-clean-100", "transcript", "raw")

dev_x_dir = os.path.join("dev-clean", "mfcc")
dev_y_dir = os.path.join("dev-clean", "transcript", "raw")

test_x_dir = os.path.join("test-clean", "mfcc")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StepTeacherForcingScheduler:
    def __init__(self, model, step_size=0.05, min=0.5):
        self._step_size = step_size
        self._min = min
        self._model = model

    def step(self):
        current_rf_rate = self._model.tf_rate
        self._model.tf_rate = max(self._min, current_rf_rate - self._step_size)


def get_labeled_data_loader(data_root, x_dir, y_dir, **kwargs):
    x_dir = os.path.join(data_root, x_dir)
    y_dir = os.path.join(data_root, y_dir)
    dataset = LabeledDataset(x_dir, y_dir)
    data_loader = DataLoader(dataset, collate_fn=LabeledDataset.collate_fn, pin_memory=True, **kwargs)
    return data_loader


# Copied from hw3p2
class LabeledDataset(Dataset):
    # load the dataset
    def __init__(self, x, y):
        # X and y are the directories containing training data and labelkk
        x_file_list = sorted([os.path.join(x, p) for p in os.listdir(x)])
        y_file_list = sorted([os.path.join(y, p) for p in os.listdir(y)])

        self.X = [np.load(p, allow_pickle=True) for p in x_file_list]
        self.Y = [np.load(p, allow_pickle=True) for p in y_file_list]
        self.seq_lengths = [x.shape[0] for x in self.X]
        self.target_lengths = [y.shape[0] for y in self.Y]

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    # get row item at some index
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        # Transform string to integer encoding
        y = np.array([VOCAB.index(c) for c in y])
        seq_length = self.seq_lengths[index]
        target_length = self.target_lengths[index]

        return (x, y), (seq_length, target_length)

    @staticmethod
    def collate_fn(batch):
        batch_x = [torch.tensor(x) for (x, _), _ in batch]
        batch_y = [torch.tensor(y, dtype=torch.long) for (_, y), _ in batch]
        batch_seq_lengths = [l for _, (l, _) in batch]
        batch_target_lengths = [t for _, (_, t) in batch]

        # Pad variable length sequences
        padded_x = pad_sequence(batch_x, batch_first=True)
        padded_y = pad_sequence(batch_y, batch_first=True)
        return (padded_x, padded_y), (batch_seq_lengths, batch_target_lengths)


class UnlabeledDataset(Dataset):
    def __init__(self, x):
        x_files = [os.path.join(x, p) for p in os.listdir(x)]
        self.X = [np.load(p, allow_pickle=True) for p in x_files]
        self.seq_lengths = [x.shape[0] for x in self.X]

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        seq_length = self.seq_lengths[index]
        return x, seq_length

    @staticmethod
    def collate_fn(batch):
        batch_x = [torch.tensor(x) for (x, _) in batch]
        batch_seq_lengths = [l for (_, l) in batch]
        return pad_sequence(batch_x, batch_first=True), batch_seq_lengths


def train_epoch(training_loader, model, criterion, optimizer, scaler, current_epoch, scheduler=None, tf_scheduler=None):
    total_training_loss = 0.0
    total_samples = 0
    total_batches = len(training_loader)
    model.train()
    b = 0
    for (batch_x, batch_y), (batch_seq_lengths, batch_target_lengths) in tqdm(training_loader):
        batch_size = batch_y.shape[0]
        model.to(device)
        optimizer.zero_grad()
        loss = labeled_forward(model, criterion, batch_x, batch_y, batch_seq_lengths, batch_target_lengths)
        total_training_loss += (float(loss) * batch_size)
        total_samples += batch_size
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            current_epoch = current_epoch + b / total_batches
            scheduler.step(current_epoch)
        if tf_scheduler is not None:
            tf_scheduler.step()
        b += 1

    training_loss = "Training loss ", total_training_loss / total_samples
    print(training_loss)


def labeled_forward(model, criterion, batch_x, batch_y, batch_seq_lengths, batch_target_lengths, validation_mode=False):

    with torch.cuda.amp.autocast():
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if validation_mode:
            output_logits = model.forward(batch_x, batch_seq_lengths)
        else:
            output_logits = model.forward(batch_x, batch_seq_lengths, batch_y)

        # We don't include <sos> when compute the loss
        batch_target_lengths = [l - 1 for l in batch_target_lengths]
        packed_logits = pack_padded_sequence(
            output_logits,
            torch.tensor(batch_target_lengths),
            batch_first=True,
            enforce_sorted=False
        )
        packed_targets = pack_padded_sequence(
            batch_y[:, 1:],
            torch.tensor(batch_target_lengths),
            batch_first=True,
            enforce_sorted=False
        )
        loss = criterion(packed_logits.data, packed_targets.data)
    return loss


def validate(model: torch.nn.Module, dev_loader):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for (batch_x, batch_y), (batch_seq_lengths, batch_target_lengths) in tqdm(dev_loader):
            batch_size = batch_y.shape[0]
            loss = labeled_forward(model, criterion, batch_x, batch_y, batch_seq_lengths, batch_target_lengths, True)
            total_loss = total_loss + batch_size * float(loss)
            total_samples += batch_size

    return total_loss / total_samples


def train_las(params: dict):
    model = LAS(
        params['char_embedding_size'],
        params['seq_embedding_size'],
        len(VOCAB),
        params['plstm_layers'],
        params['tf_rate'],
        params["encoder_dropout"],
        params["frequency_mask"],
        params["time_mask"]
    )
    print(params)
    print(model)
    n_epochs = params["n_epochs"]
    n_pretraining_epochs = params["n_pretraining_epochs"]
    data_root = params["data_root"]
    num_workers = params["num_dataloader_workers"]
    training_loader = get_labeled_data_loader(
        data_root,
        training_x_dir,
        training_y_dir,
        shuffle=True,
        batch_size=params["training_batch_size"],
        num_workers=num_workers
    )
    val_loader = get_labeled_data_loader(
        data_root,
        dev_x_dir,
        dev_y_dir,
        shuffle=False,
        batch_size=params["validation_batch_size"],
        num_workers=num_workers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    tf_scheduler = StepTeacherForcingScheduler(model, params["tf_step_size"])

    pretrain(model, training_loader, n_pretraining_epochs)

    for epoch in range(n_epochs):
        train_epoch(training_loader, model, criterion, optimizer, scaler, epoch)
        val_loss = validate(model, val_loader)
        tf_scheduler.step()
        print(f"Validation loss: {val_loss}")


def pretrain_epoch(model, training_loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for  (batch_x, _), (x_lengths, _) in tqdm(training_loader):
        model.to(device)
        batch_size = batch_x.shape[0]
        total_samples += batch_size
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        batch_x_hat = model.self_decoder_forward(batch_x, x_lengths)
        packed_batch_x = pack_padded_sequence(batch_x, x_lengths, True, False)
        packed_batch_x_hat = pack_padded_sequence(batch_x_hat, x_lengths, True, False)
        with torch.cuda.amp.autocast():
            loss = criterion(packed_batch_x.data, packed_batch_x_hat.data)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += (float(loss) * batch_size)

    training_loss = "Pre-training loss ", total_loss / total_samples
    print(training_loss)


def pretrain(model, training_loader, n_epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(n_epochs):
        pretrain_epoch(model, training_loader, criterion, optimizer, scaler)


if __name__ == "__main__":
    # The input size is typically (batch_size, max_seq_length, 15)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--n_pretraining_epochs", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--num_dataloader_workers", type=int, default=2)
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--char_embedding_size", type=int, default=256)
    parser.add_argument("--seq_embedding_size", type=int, default=32)
    parser.add_argument("--plstm_layers", type=int, default=3)
    parser.add_argument("--encoder_dropout", type=float, default=0.5)
    parser.add_argument("--decoder_dropout", type=float, default=0.5)
    parser.add_argument("--time_mask", type=int, default=30)
    parser.add_argument("--frequency_mask", type=int, default=3)
    parser.add_argument("--tf_rate", type=float, default=1.0)
    parser.add_argument("--tf_step_size", type=float, default=0.025)
    args = parser.parse_args()
    train_las(vars(args))
    print("done")
