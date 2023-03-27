import torch
import numpy as np
from phonetics import SOS_TOKEN, EOS_TOKEN
from tqdm import tqdm
from hw4p2_train import test_x_dir, device, UnlabeledDataset
from las import LAS
from hw4p2_train import get_unlabeled_dataloader, random_search
from phonetics import VOCAB




def beam_search(model: LAS, data_loader, beam_width):
    pass






def inference(params):
    model_checkpoint = params["model_path"]
    model = LAS(
        params["embedding_size"],
        params["context_size"],
        len(VOCAB),
        plstm_layers=params["plstm_layers"],
        teacher_force_rate=1,
        encoder_dropout=0.5,
        decoder_dropout=0.5,
        freq_mask=0,
        time_mask=0
    )
    dataloader = get_unlabeled_dataloader(
        params["data_root"],
        test_x_dir,
        batch_size=params["batch_size"],
        num_workers=4,
        shuffle=False
    )

    searching_method = params["searching_method"]
    model.load_state_dict(torch.load(model_checkpoint))
    if searching_method == "random":
        predicted_strings = random_search(model, dataloader, params["n_runs"])
    else:
        raise NotImplementedError()

    print(predicted_strings)
    np.save("predicton", predicted_strings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("mfcc_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--searching_method", type=str, default="random")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of random runs if searching method is random")
    parser.add_argument("--num_dataloader_workers", type=int, default=2)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--context_size", type=int, default=512)
    parser.add_argument("--plstm_layers", type=int, default=3)
    args = parser.parse_args()
    inference(vars(args))
