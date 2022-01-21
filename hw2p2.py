import os
import torch
import torchvision
import torch.nn.functional as nf
import typing as t
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomErasing, Compose, RandomPerspective

train_transforms = Compose([ToTensor(), RandomErasing(), RandomHorizontalFlip()])
val_transforms = Compose([ToTensor()])

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "hw2p2_data")


class VerificationDataSet(torch.utils.data.Dataset):
    """
    For face verification task. Each data point contains two images and
    one label indicating whether the two are the same person.
    """
    def __init__(self, path: str, with_label=True):
        self.image_paries: t.List[t.Tuple[str, str]] = []
        self.labels = []
        self.root_path: str = path
        with open(path) as f:
            for line in f.readlines():
                if with_label:
                    image1, image2, label = line.split(" ")
                else:
                    image1, image2 = line.split(" ")
                    label = line
                self.image_paries.append((image1, image2))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paries)

    def __getitem__(self, index):
        image1_path = os.path.join(self.root_path, self.image_paries[index][0])
        image2_path = os.path.join(self.root_path, self.image_paries[index][1])
        transformer = torchvision.transforms.ToTensor()
        image_1 = transformer(Image.open(image1_path))
        image_2 = transformer(Image.open(image2_path))
        label = self.labels[index]
        return (image_1, image_2), label


# For the classfication task
training_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train_data"),
                                                 transform=train_transforms)
validation_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "val_data"),
                                                   transform=val_transforms)

test_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test_data"),
                                             transform=torchvision.transforms.ToTensor())

device = torch.device("cuda")

# For the verification task
verification_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "verification_data"))


class SimpleResNetBlock(nn.Module):
    def __init__(self, channel_size, stride):
        super(SimpleResNetBlock, self).__init__()
        self.conv = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(channel_size)
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            # To assure that shortcut output has the same shape as ordinary path, need to downsample when stride > 1
            self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.relu(out + self.shortcut(x))


class ResidualBlock(nn.Module):
    """
    A size customizable residual block from recitation slides.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
        )
        # Keep the input size and output size match
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return nf.relu(out + shortcut)


class ResidualBlock3(nn.Module):
    """
    A resnet has 3 internal conv layers
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super(ResidualBlock3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels // 4),
            nn.ReLU(),
            nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels // 4),
            nn.ReLU(),
            nn.Conv2d(output_channels // 4, output_channels, kernel_size=1, stride=1)
        )

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return nf.relu(out + shortcut)


class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super(ResNet50, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),  # (64, 14, 14)
            ResidualBlock3(64, 256, 3), # (256, 14, 14)
            ResidualBlock3(256, 256, 3),  # (256, 14, 14)
            ResidualBlock3(256, 256, 3),  # (256, 14, 14)
            ResidualBlock3(256, 512, 3, 2),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 1024, 3, 2),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 2048, 3, 2),  # (2048, 14, 14)
            ResidualBlock3(2048, 2048, 3),  # (2048, 14, 14)
            ResidualBlock3(2048, 2048, 3),  # (2048, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1)),  #(2048, 1, 1)
            nn.Flatten(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ResNet18(nn.Module):
    """
    An 18-layer ResNet from recitation slides.
    """
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3, True),  # (128, 14, 14)
            ResidualBlock(128, 128, 3),  # (128, 14, 14)
            ResidualBlock(128, 256, 3, True),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 512, 3, True),  # (512, 14, 14)
            ResidualBlock(512, 512, 3),  # (512, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ResNet34(nn.Module):

    def __init__(self, n_classes):
        super(ResNet34, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=6),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3, 2),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 256, 3, 2),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 512, 3, 2),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ResNet34NoDownSample(nn.Module):
    def __init__(self, n_classes, embedding_dim=64):
        super(ResNet34NoDownSample, self).__init__()
        self.embedding_dim=embedding_dim
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
        )
        self.linear_out = nn.Linear(1000, n_classes)
        self.linear_embed = nn.Linear(1000, self.embedding_dim)

    def forward(self, x, return_embedding=False):
        out = self.layers(x)
        if return_embedding:
            return self.linear_out(out), nf.relu(self.linear_embed(out))
        else:
            return self.linear_out(out)


class ResNet34NoDownSampleSmallKernel(nn.Module):
    def __init__(self, n_classes):
        super(ResNet34NoDownSampleSmallKernel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ResNet34NoDownSampleLargerFC(nn.Module):
    def __init__(self, n_classes):
        super(ResNet34NoDownSampleLargerFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


# Construct model for the training process
class ClassificationNetwork(torch.nn.Module):

    def __init__(self, in_channels, num_classes, embedding_dim):
        super(ClassificationNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SimpleResNetBlock(64, stride=1),
            SimpleResNetBlock(64, stride=1),
            SimpleResNetBlock(64, stride=1),
            SimpleResNetBlock(64, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.linear = nn.Linear(64, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_out = nn.Linear(64, num_classes)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.relu(self.linear(embedding))
        output = self.linear_out(embedding)
        if return_embedding:
            return embedding_out, output
        return output


class ConvNetClassifier1(nn.Module):
    def __init__(self, n_classes):
        super(ConvNetClassifier1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 32, 32)
            nn.Conv2d(32, 32, 3, 1),  # (32, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 15, 15)
            nn.Conv2d(32, 128, 5, 1),  # (128, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 5, 5)
            nn.Flatten(),   # (128 * 5 * 5)
            nn.Linear(128 * 5 * 5, 4096),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_class=10, num_feature=2):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.num_feature = num_feature
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


def train_with_closeness(training_set, val_set, model, l_criterion, c_criterion, n_epochs,
                         lr_l, lr_c, weight_decay, closeness_ratio):
    training_dataloader = DataLoader(training_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader =  DataLoader(val_set, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
    ver_dataloader = DataLoader(verification_data, batch_size=1, shuffle=False)
    label_optimizer = torch.optim.SGD(model.parameters(), lr=lr_l, weight_decay=weight_decay, momentum=0.9)
    closeness_optimizer = torch.optim.SGD(c_criterion.parameters(), lr=lr_c, momentum=0.9)
    scheduler_l = ReduceLROnPlateau(label_optimizer, mode='min', factor=0.1, patience=1, cooldown=2)
    scheduler_c = ReduceLROnPlateau(closeness_optimizer, mode='min', factor=0.1, patience=1, cooldown=2)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        center_loss = 0.0
        label_loss = 0.0
        for batch_num, (x, y) in enumerate(training_dataloader):
            x = x.to(device)
            y = y.to(device)
            label_optimizer.zero_grad()
            closeness_optimizer.zero_grad()
            out, embedding = model(x, return_embedding=True)
            l_loss = l_criterion(out, y)
            c_loss = c_criterion(embedding)
            loss = l_loss + closeness_ratio * c_loss
            loss.backward()
            label_optimizer.step()
            closeness_optimizer.step()
            center_loss += c_loss.item()
            label_loss += l_loss.item()
            if batch_num % 100 == 99:
                print(f"Epoch: {e}, batch: {batch_num}, training_loss: {label_loss / 100}, center_loss: {center_loss / 100}")
                center_loss = 0.0
                label_loss = 0.0




def train(training_set, val_set, model: nn.Module, criterion, n_epochs, lr, weight_decay):
    train_dataloader = DataLoader(training_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, cooldown=2)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        avg_loss = 0.0
        for batch_num, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            # When use optimizer.zero_grad(set_to_none=True), the loss doesn't decrease at all, why?
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if batch_num % 100 == 99:
                print(f"Epoch: {e}, batch: {batch_num}, training_loss: {avg_loss / 100}")
                avg_loss = 0.0

        # Validation per epoch
        val_loss = validate(model, val_set, criterion=criterion)
        scheduler.step(val_loss)
        print(f"Validation loss at epoch {e}: {val_loss}")
        print(f"Validation accuracy at epoch {e}: {accuracy_score(np.concatenate(truth), np.concatenate(predictions))}")

    return model


def validate(model, data_set, criterion):
    val_dataloader = DataLoader(data_set, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
    model.eval()
    truth = []
    predictions = []
    val_loss = 0.0
    with torch.no_grad():
        for batch_num, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            val_loss += (criterion(outputs, y).cpu().item() * len(y))
            truth.append(y.cpu().numpy())
            predictions.append(outputs.cpu().numpy().argmax(axis=1))
    val_loss = val_loss / len(data_set)
    return val_loss


def test(model):
    classes = training_data.classes
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    result = {}
    with torch.no_grad():
        model.eval()
        model.to(device)
        for i, image in enumerate(test_data_loader):
            x = image[0].to(device)
            label = torch.argmax(model(x), dim=1)
            file_name = os.path.basename(test_data.imgs[i][0])
            result[file_name] = classes[label.item()]
    return result


def make_classification_submission(model):
    # To check model is correct
    result = test(model)
    submission = pd.DataFrame.from_records(list(result.items()), columns=['id', 'label'])
    submission.to_csv("classification_submission.csv", index=False)


def run_train():
    n_classes = len(training_data.classes)
    training_criterion = nn.CrossEntropyLoss()
    # Use Centor loss
    closeness_criterion = CenterLoss(num_class=n_classes, num_feature=64)

    # model = ClassificationNetwork(3, n_classes, 32)
    # model = ConvNetClassifier1(n_classes)    # Best validation score 0.26
    # model = ResNet18(n_classes)    # Best validation score 0.6
    # model = ResNet34(n_classes)    # Best validation score is 0.59 around epoch 20 with weight decay 1e-5
    # model = ResNet34NoDownSample(n_classes) # 0.7 with weight decay 1e-5
    # model = ResNet34NoDownSample(n_classes)  # 0.82 with weight decay 1e-3
    # model = ResNet34NoDownSample(n_classes)  # Not working with weight decay 1e-2
    # model = ResNet34NoDownSample(n_classes)   # 0.76 with weight decay 2e-3
    # model = ResNet34NoDownSample(n_classes)    # 0.829 with weight_decay 5e-4
    # model = ResNet34NoDownSampleLargerFC(n_classes) # 0.71 with weigt_decay 5e-4
    # model = ResNet34NoDownSample(n_classes)  # 0.845 with weight_decay 5e-4 and transformer Horizontal flip and random erasing.
    # model = ResNet34NoDownSample(n_classes) # 0.8125 with weight_decay 5e-4 and transformer HorizonbtalFlip RandomErasing RandomPerspective 
    # model = ResNet34NoDownSample(n_classes)  # 0.8425 with weight decay 5e-4, HorizontalFlip, RandomErasing, lr=0.2
    model = ResNet34NoDownSampleSmallKernel(n_classes)

    n_epochs = 50
    trained_model = train(training_data, validation_data, model, training_criterion, n_epochs, lr=0.1, weight_decay=5e-4)
    torch.save(trained_model.state_dict(), "saved_model")
    make_classification_submission(model)


def run_submission():
    model = ResNet34NoDownSample(4000)
    model.load_state_dict(torch.load("saved_model"))
    make_classification_submission(model)


if __name__ == "__main__":
    run_train()
