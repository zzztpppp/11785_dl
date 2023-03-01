import argparse
from hw4p2 import (
    get_labeled_data_loader,
    training_x_dir,
    training_y_dir,
    dev_x_dir,
    dev_y_dir,
    VOCAB,
    device
)
import torch
from tqdm import tqdm
from las import Listener, SelfDecoder
from torch.nn.utils.rnn import pack_padded_sequence


class PretrainerModel(torch.nn.Module):
    def __init__(self, trainer, trainee):
        super().__init__()
        self._trainer = trainer
        self._trainee = trainee

    def forward(self, batch_x, x_lengths):
        seq_embeddings, _ = self._trainee.forward(batch_x, x_lengths)
        seq_x_recovered = self._trainer.forward(seq_embeddings.transpose(1, 2)).transpose(1, 2)
        return seq_x_recovered


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
        batch_x_hat = model.forward(batch_x, x_lengths)
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


def pretrain(trainee, trainer, training_dataloader, n_epochs):
    pretraining_model = PretrainerModel(trainer, trainee)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(pretraining_model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(n_epochs):
        pretrain_epoch(pretraining_model, training_dataloader, criterion, optimizer, scaler)
        lr_scheduler.step(epoch)


def pretrain_listener(params):
    n_epochs = params["n_epochs"]
    data_root = params["data_root"]
    num_workers = params["num_dataloader_workers"]
    n_plstm_layers = params["plstm_layers"]
    seq_embedding_size = params["seq_embedding_size"]
    training_loader = get_labeled_data_loader(
        data_root,
        training_x_dir,
        training_y_dir,
        shuffle=True,
        batch_size=params["training_batch_size"],
        num_workers=num_workers
    )
    listener_model = Listener(
        15,
        seq_embedding_size,
        n_plstm_layers,
        params["encoder_dropout"]
    )
    listener_out_size = seq_embedding_size * (2 ** n_plstm_layers)

    self_decoder_model = SelfDecoder(listener_out_size, 15, n_plstm_layers)
    pretrain(trainee=listener_model, trainer=self_decoder_model,
             training_dataloader=training_loader, n_epochs=n_epochs)

    torch.save(listener_model.state_dict(), "pretrained_listener")


if __name__ == "__main__":
    # The input size is typically (batch_size, max_seq_length, 15)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--num_dataloader_workers", type=int, default=2)
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--seq_embedding_size", type=int, default=32)
    parser.add_argument("--plstm_layers", type=int, default=3)
    parser.add_argument("--encoder_dropout", type=float, default=0.5)
    parser.add_argument("--time_mask", type=int, default=30)
    parser.add_argument("--frequency_mask", type=int, default=3)
    args = parser.parse_args()
    pretrain_listener(vars(args))
    print("done")
