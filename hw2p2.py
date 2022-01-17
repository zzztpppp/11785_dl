import os
import torch
import torchvision
import torch.nn.functional as nf
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "hw2p2_data")

# For the classfication task
training_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train_data"),
                                                 transform=torchvision.transforms.ToTensor())
validation_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "val_data"),
                                                   transform=torchvision.transforms.ToTensor())
# test_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test_data"))
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
    def __init__(self, input_channels, output_channels, kernel_size, convolutional_shortcut=False):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
        )
        # Keep the input size and output size match
        if convolutional_shortcut:
            self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1)
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
    def __init__(self, input_channels, output_channels, kernel_size, convolutional_shortcut=False):
        super(ResidualBlock3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels // 4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels // 4),
            nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels // 4),
            nn.Conv2d(output_channels // 4, output_channels, kernel_size=1, stride=1)
        )

        if convolutional_shortcut:
            self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
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
            ResidualBlock3(64, 256, 3, True), # (256, 14, 14)
            ResidualBlock3(256, 256, 3),  # (256, 14, 14)
            ResidualBlock3(256, 256, 3),  # (256, 14, 14)
            ResidualBlock3(256, 512, 3, True),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 512, 3),  # (512, 14, 14)
            ResidualBlock3(512, 1024, 3, True),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 1024, 3),  # (1024, 14, 14)
            ResidualBlock3(1024, 2048, 3, True),  # (2048, 14, 14)
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
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3, True),  # (128, 14, 14)
            ResidualBlock(128, 128, 3),  # (128, 14, 14)
            ResidualBlock(128, 128, 3),  # (128, 14, 14)
            ResidualBlock(128, 128, 3),  # (128, 14, 14)
            ResidualBlock(128, 256, 3, True),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 512, 3, True),  # (512, 14, 14)
            ResidualBlock(512, 512, 3),  # (512, 14, 14)
            ResidualBlock(512, 512, 3),  # (512, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_classes)
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


def train(training_set, val_set, model: nn.Module, criterion, n_epochs, lr, weight_decay):
    train_dataloader = DataLoader(training_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
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
        val_loss = val_loss / len(val_set)
        scheduler.step(val_loss)
        print(f"Validation loss at epoch {e}: {val_loss}")
        print(f"Validation accuracy at epoch {e}: {accuracy_score(np.concatenate(truth), np.concatenate(predictions))}")

    return model


def eval(model, test_data):
    model.eval()
    with torch.no_grad():
        pass



def run_train():
    n_classes = len(training_data.classes)
    training_criterion = nn.CrossEntropyLoss()
    # model = ClassificationNetwork(3, n_classes, 32)
    # model = ConvNetClassifier1(n_classes)    # Best validation score 0.26
    # model = ResNet18(n_classes)    # Best validation score 0.6
    # model = ResNet34(n_classes)    # Best validation score is 0.79 around epoch 20
    model = ResNet50(n_classes)
    n_epochs = 20
    trained_model = train(training_data, validation_data, model, training_criterion, n_epochs, lr=0.1, weight_decay=1e-3)
    torch.save(trained_model.state_dict(), "saved_model")


if __name__ == "__main__":
    run_train()
