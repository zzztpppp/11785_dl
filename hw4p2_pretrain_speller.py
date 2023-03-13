# Pretrain the speller like a language model
import torch
import os
import numpy as np
from phonetics import VOCAB
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from las import Speller
from torch.nn.utils.rnn import pad_sequence
from hw4p2 import training_y_dir, dev_x_dir, device
from las import Speller


class PretrainDataset(Dataset):
    def __init__(self, dir):
        data_file_list = sorted([os.path.join(dir, p) for p in os.listdir(dir)])
        self.data = [np.load(p, allow_pickle=True) for p in data_file_list]
        self.seq_lengths = [x.shape[0] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        length = self.seq_lengths[index]
        data = np.array([VOCAB.index(c) for c in data])
        return (data[:1], data[1:]), length - 1

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


def pretrain_epoch(model, training_loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0
    model.to(device)
    for (batch_x, batch_y), batch_lengths in tqdm(training_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        model.forward(batch_x)



def pretrain(params):
    data_root = params["data_root"]
    num_workers = params["num_dataloader_workers"]
    batch_size = params["training_batch_size"]
    train_loader = get_dataloader(data_root, training_y_dir, num_workers=num_workers, batch_size=batch_size)
    speller = Speller(
        embedding_size=params["embedding_size"],
        output_size=len(VOCAB)
    )
    n_epochs = params["n_epochs"]
    for epoch in range(n_epochs):
        pass
