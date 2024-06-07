import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchsummary import summary
import torchvision  # This library is used for image-based operations (Augmentations)
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
import glob
import wandb
from models import ResNet18, ResNet34, ConvNextT, ArcMarginProduct
from timm.data import create_transform

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


DATA_DIR = 'classification_data'  # TODO: Path where you have downloaded the data
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "dev")
TEST_DIR = os.path.join(DATA_DIR, "test")


# Transforms using torchvision - Refer https://pytorch.org/vision/stable/transforms.html
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def build_training_transforms():
    # From https://github.com/facebookresearch/ConvNeXt/blob/main/datasets.py#L50
    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=RGB_MEAN,
        std=RGB_STD,
    )
    return transform


# You can do this with ImageFolder as well, but it requires some tweaking
class ClassificationTestDataset(Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


def train(model, dataloader, optimizer, criterion, config, epoch, training_steps_per_epoch,
          lr_scheduled_values=None, metric=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler()  # Good news. We have FP16 (Mixed precision training) implemented for you
    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss = 0
    loss_type = config["loss"]
    for step, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        # Adjust learning rate with scheduled values
        global_steps = step + training_steps_per_epoch * epoch
        if lr_scheduled_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduled_values[global_steps] * config["layer_lr_scale"][i]

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!

            if loss_type == "arcface":
                embeddings = model.forward_feat(images)
                outputs = metric.forward(embeddings, labels)
            elif loss_type == "softmax":
                outputs = model(images)
            else:
                raise ValueError(f"Unknown loss {loss_type}")

            loss = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size'] * (step + 1))),
            loss="{:.04f}".format(float(total_loss / (step + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        # TODO? Depending on your choice of scheduler,
        # You may want to call some schdulers inside the train function. What are these?

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion, config):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        # Move images to device
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size'] * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss


def experiment():
    config = {
        'batch_size': 128,  # Increase this if your GPU can handle it
        'lr': 0.1,
        "layer_lr_scale": [0.001, 1],
        "min_lr": 1e-8,
        "weight_decay": 2e-4,
        "warmup_epochs": 10,
        'epochs': 100,
        "arch": "resnet",
        "label_smoothing": 0.1,
        "optimizer": "sgd",
        "m": 0.5,
        "s": 64,
        "cls_dropout": False,
        "loss": "arcface",
        "resume_from_ckpt": r"D:\code\cmu11785\best_classification.pth"

        # 20 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
        # Include other parameters as needed.
    }
    wandb.login(key="d9064f7e7a933b775df41f6fbd3d2ed5a1050d27")

    # Most torchvision transforms are done on PIL images. So you convert it into a tensor at the end with ToTensor()
    # But there are some transforms which are performed after ToTensor() : e.g - Normalization
    # Normalization Tip - Do not blindly use normalization that is not suitable for this dataset

    # You should NOT have data augmentation on the validation set. Why?
    valid_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    print("Device: ", DEVICE)
    if config["arch"] == "resnet":
        model = ResNet34(7001, config["cls_dropout"])
    elif config["arch"] == "convnext":
        model = ConvNextT(3, 7001)
    else:
        raise ValueError(f"Unknown arch {config['arch']}")

    summary(model, (3, 224, 224))
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=build_training_transforms())
    valid_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=valid_transforms)
    run = wandb.init(
        name="weight-decay-experiment",  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Allows reinitalizing runs when you re-run this cell
        # id="izphf5f0",### Insert specific run id here if you want to resume a previous run
        # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw2p2-ablations",  ### Project should be created in your wandb account
        config=config  ### Wandb Config for your run
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # TODO: What loss do you need for a multi class classification problem?
    optimizer_name = config["optimizer"]
    total_parameters = [{"params": model.parameters()}]
    metric = None
    if config["loss"] == "arcface":
        metric = ArcMarginProduct(1000, 7001, easy_margin=True, m=config["m"], s=config["s"])
        metric.to(DEVICE)
        metric_params = {"params": metric.parameters()}
        # Use the same optimizer config
        total_parameters.append(metric_params)

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(total_parameters, lr=config['lr'], momentum=0.9, weight_decay=config["weight_decay"])
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(total_parameters, lr=config['lr'], weight_decay=config["weight_decay"])
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(total_parameters, lr=config['lr'], weight_decay=config["weight_decay"])
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

    if wandb.run.resumed:
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if config["resume_from_ckpt"]:
        print(f"Resume from {config['resume_from_ckpt']}")
        checkpoint = torch.load(config["resume_from_ckpt"])
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Verification dataset
    unknown_dev, unknown_test, known_images, known_paths, similarity_metric = get_verification_data()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scheduler_values = cosine_scheduler(
        config["lr"],
        config["min_lr"],
        config["epochs"],
        len(train_loader),
        config["warmup_epochs"],
    )
    # You can try ReduceLRonPlateau, StepLR, MultistepLR, CosineAnnealing, etc.
    # It is useful only in the case of compatible GPUs such as T4/V100
    best_valacc = 0.0
    best_veracc = 0.0
    for epoch in range(config['epochs']):

        curr_lr = float(optimizer.param_groups[0]['lr'])

        train_acc, train_loss = train(model, train_loader, optimizer, criterion,
                                      config=config, epoch=epoch, training_steps_per_epoch=len(train_loader),
                                      lr_scheduled_values=scheduler_values, metric=metric)

        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_acc,
            train_loss,
            curr_lr))

        val_acc, val_loss = validate(model, valid_loader, criterion, config=config)
        # scheduler.step(val_acc)
        verification_acc = eval_verification(
            unknown_images=unknown_dev, known_images=known_images,
            known_paths=known_paths, similarity=similarity_metric,
            model=model, mode="val",
        )

        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

        wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
                   'validation_loss': val_loss, "learning_Rate": curr_lr, "verification_acc": verification_acc})

        # If you are using a scheduler in your train function within your iteration loop, you may want to log
        # your learning rate differently

        # #Save model in drive location if val_acc is better than best recorded val_acc
        if val_acc >= best_valacc:
            # path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
            print("Saving model")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './checkpoint.pth')
            best_valacc = val_acc
            wandb.save('checkpoint.pth')
            generate_outputs(model, valid_transforms)
            # You may find it interesting to exlplore Wandb Artifcats to version your models
        if verification_acc >= best_veracc:
            print(f"Saving model for best verification acc {verification_acc}")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './ver_checkpoint.pth')
            best_veracc = verification_acc
            eval_verification(unknown_images=unknown_test, known_images=known_images, known_paths=known_paths,
                              similarity=similarity_metric, model=model, mode="test")
            wandb.save('ver_checkpoint.pth')

    run.finish()



def generate_outputs_from_checkpoint(checkpoint_path):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])
    checkpoint = torch.load(checkpoint_path)
    model = ResNet34(7001)
    model = model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    generate_outputs(model, transforms)


def generate_outputs(model, transforms):
    dataset = ClassificationTestDataset(TEST_DIR,
                                             transforms=transforms)  # Why are we using val_transforms for Test Data?
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                              drop_last=False, num_workers=2)
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    test_results = []

    for i, (images) in enumerate(dataloader):
        # TODO: Finish predicting on the test set.
        images = images.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
        test_results.extend(outputs)

        batch_bar.update()

    batch_bar.close()
    with open("classification_early_submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(dataset)):
            f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", test_results[i]))
    return test_results


def get_verification_data():
    # This obtains the list of known identities from the known folder
    known_regex = "verification_data/known/*/*"
    known_regex = r"D:\code\cmu11785\hw_kaggle\fall23_hw2p2\verification_data\known\*\*"
    known_paths = [i.split('\\')[-2] for i in sorted(glob.glob(known_regex))]

    # Obtain a list of images from unknown folders
    unknown_dev_regex = r"D:\code\cmu11785\hw_kaggle\fall23_hw2p2\verification_data\unknown_dev\*"
    unknown_test_regex = r"D:\code\cmu11785\hw_kaggle\fall23_hw2p2\verification_data\unknown_test\*"

    # We load the images from known and unknown folders
    unknown_dev_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_dev_regex)))]
    unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_test_regex)))]
    known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

    # Why do you need only ToTensor() here?
    transforms = torchvision.transforms.v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ]
    )

    unknown_dev_images = torch.stack([transforms(x) for x in unknown_dev_images])
    unknown_test_images = torch.stack([transforms(x) for x in unknown_test_images])
    known_images = torch.stack([transforms(y) for y in known_images])
    # Print your shapes here to understand what we have done

    # You can use other similarity metrics like Euclidean Distance if you wish
    similarity_metric = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # ckpt = torch.load("checkpoint.pth")
    # model = ResNet34(7001)
    # model.load_state_dict(ckpt["model_state_dict"])
    # model.to(DEVICE)
    # eval_verification(
    #     unknown_images=unknown_test_images,
    #     known_images=known_images,
    #     known_paths=known_paths,
    #     similarity=similarity_metric,
    #     model=model,
    #     mode="test"
    # )
    return unknown_dev_images, unknown_test_images, known_images, known_paths, similarity_metric


def eval_verification(unknown_images, known_images, known_paths, model, similarity, batch_size=512, mode='val'):
    unknown_feats, known_feats = [], []

    batch_bar = tqdm(total=len(unknown_images) // batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    model.eval()

    # We load the images as batches for memory optimization and avoiding CUDA OOM errors
    for i in range(0, unknown_images.shape[0], batch_size):
        unknown_batch = unknown_images[i:i + batch_size]  # Slice a given portion upto batch_size

        with torch.no_grad():
            unknown_feat = model.forward_feat(unknown_batch.float().to(DEVICE))  # Get features from model
        unknown_feats.append(unknown_feat)
        batch_bar.update()

    batch_bar.close()

    batch_bar = tqdm(total=len(known_images) // batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)

    for i in range(0, known_images.shape[0], batch_size):
        known_batch = known_images[i:i + batch_size]
        with torch.no_grad():
            known_feat = model.forward_feat(known_batch.float().to(DEVICE))

        known_feats.append(known_feat)
        batch_bar.update()

    batch_bar.close()

    # Concatenate all the batches
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)

    similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
    # Print the inner list comprehension in a separate cell - what is really happening?

    max_similarity_values, predictions = similarity_values.max(
        0)  # Why are we doing an max here, where are the return values?
    max_similarity_values, predictions = max_similarity_values.cpu().numpy(), predictions.cpu().numpy()

    # Note that in unknown identities, there are identities without correspondence in known identities.
    # Therefore, these identities should be not similar to all the known identities, i.e. max similarity will be below a certain
    # threshold compared with those identities with correspondence.

    # In early submission, you can ignore identities without correspondence, simply taking identity with max similarity value
    # pred_id_strings = [known_paths[i] for i in predictions]  # Map argmax indices to identity strings

    # After early submission, remove the previous line and uncomment the following code

    thresholds = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    NO_CORRESPONDENCE_LABEL = 'n000000'
    best_pred_id_strings = []
    best_accuracy = 0.0
    if mode == "val":
        true_ids = pd.read_csv(r'D:\code\cmu11785\hw_kaggle\fall23_hw2p2\verification_data\verification_dev.csv')[
            'label'].tolist()
        for threshold in thresholds:
            pred_id_strings = []
            for idx, prediction in enumerate(predictions):
                if max_similarity_values[idx] < threshold: # why < ? Thank about what is your similarity metric
                    pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
                else:
                    pred_id_strings.append(known_paths[prediction])

            accuracy = accuracy_score(pred_id_strings, true_ids)
            if accuracy > best_accuracy:
                best_pred_id_strings = pred_id_strings
                best_accuracy = accuracy
            print("Verification Accuracy = {} with threshold {}".format(accuracy, threshold))
    if mode == "test":
        threshold = 0.3
        pred_id_strings = []
        for idx, prediction in enumerate(predictions):
            if max_similarity_values[idx] < threshold: # why < ? Thank about what is your similarity metric
                pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
            else:
                pred_id_strings.append(known_paths[prediction])
        with open("verification_submission.csv", "w+") as f:
            f.write("id,label\n")
            for i, pred in enumerate(pred_id_strings):
                f.write("{},{}\n".format(i, pred))

    return best_accuracy


if __name__ == "__main__":
    experiment()
