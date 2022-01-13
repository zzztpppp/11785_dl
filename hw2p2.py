import os
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "hw2p2_data")

# For the classfication task
training_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train_data"),
                                                 transform=torchvision.transforms.ToTensor())
validation_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "val_data"),
                                                   transform=torchvision.transforms.ToTensor())
# test_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test_data"))

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


def train(training_set, val_set, train_model: nn.Module, criterion, n_epochs, lr, weight_decay):
    train_dataloader = DataLoader(training_set, batch_size=128, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    for e in range(n_epochs):
        model.train()
        avg_loss = 0.0
        for batch_num, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if batch_num % 10 == 9:
                print(f"Epoch: {e}, batch: {batch_num}, training_loss: {avg_loss / 10}")
                avg_loss = 0.0

        # Validation per epoch
        train_model.eval()
        truth = []
        predictions = []
        for batch_num, (x, y) in enumerate(val_dataloader):
            outputs = model(x).numpy()
            truth.append(y.numpy())
            predictions.append(outputs.argmax(axis=1))
        print(f"Validation accuracy at epoch {e}: {accuracy_score(np.concatenate(truth), np.concatenate(predictions))}")


if __name__ == "__main__":
    n_classes = len(training_data.classes)
    training_criterion = nn.CrossEntropyLoss()
    model = ClassificationNetwork(3, n_classes, 32)
    n_epochs = 10
    train(training_data, validation_data, model, training_criterion, n_epochs, lr=0.1, weight_decay=1e-3)
