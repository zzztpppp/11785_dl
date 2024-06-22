import itertools
import os
import warnings

import numpy as np
import torch
import tqdm
# imports for decoding and distance calculation
import wandb
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

from utils import DEVICE
from models import ASRModel
from utils import VOCAB, VOCAB_MAP, calc_edit_distance, SOS_TOKEN, EOS_TOKEN, indices_to_chars
from torchsummary import summary
from speechpy.processing import cmvn
from utils import cosine_scheduler, plot_attention
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')


class SpeechDatasetME(torch.utils.data.Dataset):  # Memory efficient
    # Loades the data in get item to save RAM

    def __init__(self, root, partition="train-clean-360", transforms=None, cepstral=True):

        self.VOCAB = VOCAB
        self.cepstral = cepstral
        mfcc_dir_template = os.path.join(root, "{partition}", "mfcc")
        transcript_dir_template = os.path.join(root, "{partition}", "transcripts")
        self.transforms = transforms
        if partition == "train-clean-100" or partition == "train-clean-360":
            mfcc_dir = mfcc_dir_template.format(partition=partition)
            transcript_dir = transcript_dir_template.format(partition=partition)
            mfcc_files = [os.path.join(mfcc_dir, x) for x in sorted(os.listdir(mfcc_dir))]
            transcript_files = [os.path.join(transcript_dir, x) for x in sorted(os.listdir(transcript_dir))]
        else:
            partitions = ["train-clean-100", "train-clean-360"]
            mfcc_dir = [mfcc_dir_template.format(partition=x) for x in partitions]
            transcript_dir = [transcript_dir_template.format(partition=x) for x in partitions]

            mfcc_files = list(
                itertools.chain(
                    *[
                        [os.path.join(x, p) for p in sorted(os.listdir(x))]
                        for x in mfcc_dir
                    ]
                )
            )
            transcript_files = list(
                itertools.chain(
                    *[
                        [os.path.join(x, p) for p in sorted(os.listdir(x))]
                        for x in transcript_dir
                    ]
                )
            )

        assert len(mfcc_files) == len(transcript_files)

        self.mfcc_files = mfcc_files
        self.transcript_files = transcript_files
        self.length = len(transcript_files)
        print("Loaded file paths ME: ", partition)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        # Load the mfcc and transcripts from the mfcc and transcript paths created earlier
        mfcc = np.load(self.mfcc_files[ind], allow_pickle=True)
        transcript = np.load(self.transcript_files[ind], allow_pickle=True)

        # Normalize the mfccs and map the transcripts to integers
        if self.cepstral:
            mfcc = cmvn(mfcc, variance_normalization=True)
        transcript_mapped = [VOCAB_MAP[x] for x in transcript]

        mfcc = torch.FloatTensor(mfcc)
        transcript = torch.LongTensor(transcript_mapped)

        if self.transforms is not None:
            mfcc = self.transforms(mfcc.transpose(0, 1)).transpose(0, 1)
        return mfcc, transcript


    @staticmethod
    def collate_fn(batch):

        batch_x, batch_y, lengths_x, lengths_y = [], [], [], []

        for x, y in batch:
            # Add the mfcc, transcripts and their lengths to the lists created above
            batch_x.append(x)
            batch_y.append(y)
            lengths_x.append(len(x))
            lengths_y.append(len(y))

        # pack the mfccs and transcripts using the pad_sequence function from pytorch
        batch_x_pad = pad_sequence(
            batch_x,
            batch_first=True,
        )

        batch_y_pad = pad_sequence(
            batch_y,
            batch_first=True,
        )

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


class SpeechDatasetTest(torch.utils.data.Dataset):

    def __init__(self, root, partition, cepstral=False):

        self.mfcc_dir = os.path.join(root, "test-clean", "mfcc")
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))

        self.mfccs = []
        for i, filename in enumerate(tqdm.tqdm(self.mfcc_files)):
            mfcc = np.load(os.path.join(self.mfcc_dir, filename), allow_pickle=True)
            if cepstral:
                # Normalize the mfccs
                mfcc = cmvn(mfcc, variance_normalization=True)
            # append the mfcc to the mfcc list created earlier
            self.mfccs.append(mfcc)

        print("Loaded: ", partition)

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, ind):
        return torch.FloatTensor(self.mfccs[ind])

    @staticmethod
    def collate_fn(batch):

        batch_x, lengths_x = [], []
        for x in batch:
            # Append the mfccs and their lengths to the lists created above
            batch_x.append(x)
            lengths_x.append(len(x))

        # pack the mfccs using the pad_sequence function from pytorch
        batch_x_pad = pad_sequence(batch_x, batch_first=True)

        return batch_x_pad, torch.tensor(lengths_x)


def get_test_dataloader(data_root, cepstral):
    test_dataset = SpeechDatasetTest(
        root=data_root,
        partition='test-clean',
        cepstral=cepstral,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )
    return test_loader


def get_dataloaders(config):
    DATA_DIR = config["data_root"]
    PARTITION = config['train_dataset']
    CEPSTRAL = config['cepstral_norm']
    training_transforms = nn.Sequential(
        TimeMasking(config["time_mask_param"], p=config["time_mask_p"]),
        FrequencyMasking(config["freq_mask_param"]),
    )
    train_dataset = SpeechDatasetME(  # Or AudioDatasetME
        root=DATA_DIR,
        partition=PARTITION,
        cepstral=CEPSTRAL,
        transforms=training_transforms,
    )
    valid_dataset = SpeechDatasetME(
        root=DATA_DIR,
        partition='dev-clean',
        cepstral=CEPSTRAL
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=196,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn
    )

    test_loader = get_test_dataloader(DATA_DIR, CEPSTRAL)

    return train_loader, valid_loader, test_loader


def experiment(config):
    wandb.login(key="d9064f7e7a933b775df41f6fbd3d2ed5a1050d27")
    run = wandb.init(
        name="las",  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Allows reinitalizing runs when you re-run this cell
        # id="izphf5f0",### Insert specific run id here if you want to resume a previous run
        # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw4p2-experiment",  ### Project should be created in your wandb account
        config=config  ### Wandb Config for your run
    )

    train_loader, valid_loader, test_loader = get_dataloaders(config)

    model = ASRModel(28, config["hidden_size"], voc_size=len(VOCAB), seq_embedding_layers=config["seq_embed_layers"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    best_val_distance = 1000
    model.to(DEVICE)
    summary(model)
    steps_per_epoch = len(train_loader)
    scheduled_tf_rate = cosine_scheduler(
        config["max_tf_rate"],
        config["min_tf_rate"],
        config["n_epochs"],
        niter_per_ep=steps_per_epoch,
    )
    scheduled_lr = cosine_scheduler(
        config["max_lr"],
        config["min_lr"],
        config["n_epochs"],
        steps_per_epoch,
        warmup_epochs=1,
    )
    validation_period = 1
    for epoch in range(config["n_epochs"]):
        loss, perplexity, attention_plot = train(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            current_epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            scheduled_tf_rate=scheduled_tf_rate,
            scheduled_lr=scheduled_lr,
            gradient_norm=config["gradient_norm"],
            gumble=config["gumble"],
            hard_gumble=config["hard_gumble"],
        )
        if (epoch + 1) % validation_period == 0:
            edit_distance = validate(model, valid_loader)
            wandb.log(
                {
                    "train_loss": loss,
                    "train_perplexity": perplexity,
                    "validation_distance": edit_distance
                }
            )
            if edit_distance < best_val_distance:
                best_val_distance = edit_distance
                print(f"Saving best checkpoitn with val_distance {best_val_distance}")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "epoch": epoch,
                        "val_distance": best_val_distance,
                    },
                    "ckpt.pth"
                )
                wandb.save("best_ckpt.pth")

    plot_attention(attention_plot.mean(dim=0)[:, :, 0].detach().cpu())
    run.finish()


def train(
        model,
        dataloader,
        criterion,
        optimizer,
        scaler,
        current_epoch,
        steps_per_epoch,
        scheduled_tf_rate,
        scheduled_lr,
        gradient_norm,
        gumble,
        hard_gumble,
):
    model.train()
    batch_bar = tqdm.tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train')

    running_loss = 0.0
    running_perplexity = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):
        optimizer.zero_grad()
        global_steps = current_epoch * steps_per_epoch + i
        teacher_forcing_rate = scheduled_tf_rate[global_steps]

        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduled_lr[global_steps]

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly
        with torch.cuda.amp.autocast():
            raw_predictions, attention_plot = model(x, lx, y=y, tf_rate=teacher_forcing_rate,
                                                    gumble=gumble, hard_gumble=hard_gumble)
            # Predictions are of Shape (batch_size, timesteps, vocab_size).
            # Transcripts are of shape (batch_size, timesteps) Which means that you have batch_size amount of batches with timestep number of tokens.
            # So in total, you have batch_size*timesteps amount of characters.
            # Similarly, in predictions, you have batch_size*timesteps amount of probability distributions.
            # How do you need to modify transcipts and predictions so that you can calculate the CrossEntropyLoss? Hint: Use Reshape/View and read the docs
            # Also we recommend you plot the attention weights, you should get convergence in around 10 epochs, if not, there could be something wrong with
            # your implementation
            loss = criterion(
                pack_padded_sequence(raw_predictions, lengths=ly, batch_first=True, enforce_sorted=False).data,
                pack_padded_sequence(y, ly, batch_first=True, enforce_sorted=False).data
            )

            perplexity = torch.exp(loss)  # Perplexity is defined the exponential of the loss

            running_loss += loss.item()
            running_perplexity += perplexity.item()

        # Backward on the masked loss
        scaler.scale(loss).backward()

        # Optional: Use torch.nn.utils.clip_grad_norm to clip gradients to prevent them from exploding, if necessary
        # If using with mixed precision, unscale the Optimizer First before doing gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss / (i + 1)),
            perplexity="{:.04f}".format(running_perplexity / (i + 1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_rate))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


def validate(model, dataloader):
    model.eval()

    batch_bar = tqdm.tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=True, desc="Val")

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, (x, y, lx, ly) in enumerate(dataloader):
            x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

            with torch.inference_mode():
                raw_predictions, attentions = model(x, lx, y=None)

            # Greedy Decoding
            greedy_predictions = raw_predictions.argmax(dim=2)

            # Calculate Levenshtein Distance
            # running_lev_dist += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example=False)
            # You can use print_example = True for one specific index i in your batches if you want
            futures.append(executor.submit(calc_edit_distance, greedy_predictions.cpu().numpy(), y.cpu().numpy(), ly, VOCAB, print_example=False))
            # batch_bar.set_postfix(
            #     dist="{:.04f}".format(running_lev_dist / (i + 1)))
            batch_bar.update()
            del x, y, lx, ly, greedy_predictions
            torch.cuda.empty_cache()
        mean_distance = np.mean([f.result() for f in futures])
        print(f"Validation distance {mean_distance}")
    batch_bar.close()
    return mean_distance


def output_result(config, model, dataloader):
    if model is None:
        model = ASRModel(28, config["hidden_size"], len(VOCAB), config["seq_embed_layers"])
        static_dict = torch.load("ckpt.pth")["model_state_dict"]
        model.load_state_dict(static_dict)
    if dataloader is None:
        dataloader = get_test_dataloader(config["data_root"], config["cepstral_norm"])

    model.eval()
    model.to(DEVICE)
    all_predictions = []
    for i, (x, lx) in enumerate(dataloader):
        x, lx = x.to(DEVICE), lx
        # Greedy Decoding
        with torch.inference_mode():
            raw_predictions, attentions = model(x, lx, y=None)
        greedy_predictions = raw_predictions.argmax(dim=2)
        all_predictions.extend(greedy_predictions.cpu().tolist())

    all_prediction_strings = ["".join(indices_to_chars(x, VOCAB)) for x in all_predictions]
    with open("hw4p2.csv", "w+") as f:
        f.write("index,label\n")
        for i in range(len(all_prediction_strings)):
            f.write("{},{}\n".format(i, all_prediction_strings[i]))
    return all_prediction_strings


def main():
    config = dict(
        data_root=r'D:\code\cmu11785\hw_kaggle\fall23_hw4p2\data',
        train_dataset='train-clean-360',  # train-clean-100, train-clean-360, train-clean-460
        batch_size=96,
        n_epochs=100,
        cepstral_norm=True,

        # Teacher forcing
        min_tf_rate=0.3,
        max_tf_rate=1.0,

        # Data augmentation
        time_mask_param=30,
        freq_mask_param=10,
        time_mask_p=0.3,

        # Model size
        hidden_size=512,
        seq_embed_layers=2,

        # Hyper parameters
        min_lr=1e-6,
        max_lr=5e-4,
        weight_decay=5e-3,
        gradient_norm=1,

        gumble=False,
        hard_gumble=False,
    )
    experiment(config)
    # output_result(config, None, None)


if __name__ == "__main__":
    main()
