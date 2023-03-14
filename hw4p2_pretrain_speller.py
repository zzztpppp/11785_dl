# Pretrain the speller like a language model
import torch
import os
import numpy as np
import argparse
from phonetics import VOCAB
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from las import Speller
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from hw4p2_train import training_y_dir, dev_y_dir, device
from las import Speller


class PretrainDataset(Dataset):
    def __init__(self, data_dir):
        data_file_list = sorted([os.path.join(data_dir, p) for p in os.listdir(data_dir)])
        self.data = [np.load(p, allow_pickle=True) for p in data_file_list]
        self.seq_lengths = [x.shape[0] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        length = self.seq_lengths[index]
        data = np.array([VOCAB.index(c) for c in data])
        return (data[:-1], data[1:]), length - 1

    @staticmethod
    def collate_fn(batch):
        batch_x = [torch.tensor(x, dtype=torch.long) for (x, _), _ in batch]
        batch_y = [torch.tensor(y, dtype=torch.long) for (_, y), _ in batch]
        batch_lengths = [torch.tensor(l, dtype=torch.long) for _, l in batch]
        padded_x = pad_sequence(batch_x, batch_first=True)
        padded_y = pad_sequence(batch_y, batch_first=True)
        return (padded_x, padded_y), batch_lengths


def get_dataloader(data_root, path, **kwargs):
    data_path = os.path.join(data_root, path)
    dataset = PretrainDataset(data_path)
    data_loader = DataLoader(dataset, collate_fn=PretrainDataset.collate_fn, pin_memory=True, **kwargs)
    return data_loader


def pretrain_epoch(model: Speller, training_loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0
    model.to(device)
    # During pretraining, set the input context to 0s
    pseudo_context = torch.zeros(1, 1, model.context_size).to(device)
    for (batch_x, batch_y), batch_lengths in tqdm(training_loader):
        batch_size, max_seq_length = batch_x.shape
        loss = pretrain_forward(model, batch_x, batch_y, batch_lengths, criterion, pseudo_context)
        total_loss += (float(loss) * batch_size)
        total_samples += batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_loss = f"Speller pretraining loss {total_loss / total_samples}"
    print(training_loss)


def pretrain_forward(model, batch_x, batch_y, batch_lengths, criterion, pseudo_context):
    batch_size, max_seq_length = batch_x.shape
    batch_x = batch_x.to(device)  # (B, L, V)
    batch_y = batch_y.to(device)

    batch_x_chars = model.char_embedding.forward(batch_x)  # (B, L, E)
    batch_x_chars = torch.cat([batch_x_chars,
                               pseudo_context.expand(batch_size, max_seq_length, model.context_size)], dim=2)
    packed_batch_x = pack_padded_sequence(batch_x_chars, batch_lengths, batch_first=True, enforce_sorted=False)
    packed_lstm_out, _ = model.decoder.forward(packed_batch_x)
    packed_batch_y_dist = model.cdn.forward(packed_lstm_out.data)
    packed_batch_y = pack_padded_sequence(batch_y, batch_lengths, batch_first=True, enforce_sorted=False)
    loss = criterion(packed_batch_y_dist, packed_batch_y.data)
    return loss


def validate(model: Speller, validation_loader, criterion):
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_samples = 0
    pseudo_context = torch.zeros(1, 1, model.context_size).to(device)
    with torch.inference_mode():
        for (batch_x, batch_y), batch_lengths in tqdm(validation_loader):
            batch_size, max_seq_length = batch_x.shape
            loss = pretrain_forward(model, batch_x, batch_y, batch_lengths, criterion, pseudo_context)
            total_loss += (float(loss) * batch_size)
            total_samples += batch_size
    validation_loss = f"Speller pretraining validation loss {total_loss / total_samples}"
    print(validation_loss)


def pretrain(params):
    data_root = params["data_root"]
    num_workers = params["num_dataloader_workers"]
    batch_size = params["training_batch_size"]
    validation_batch_size = params["validation_batch_size"]
    train_loader = get_dataloader(data_root, training_y_dir, num_workers=num_workers, batch_size=batch_size)
    val_loader = get_dataloader(data_root, dev_y_dir, num_workers=num_workers, batch_size=validation_batch_size)
    speller = Speller(
        embedding_size=params["embedding_size"],
        output_size=len(VOCAB)
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(speller.parameters(), lr=0.01)
    n_epochs = params["n_epochs"]
    for epoch in range(n_epochs):
        pretrain_epoch(speller, train_loader, criterion, optimizer, None)
        validate(speller, validation_loader=val_loader, criterion=criterion)

    torch.save(speller.state_dict(), "pretrained_speller")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--num_dataloader_workers", type=int, default=2)
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=128)
    parser.add_argument("--embedding_size", type=int, default=512)
    args = parser.parse_args()
    pretrain(vars(args))
    print("done")
