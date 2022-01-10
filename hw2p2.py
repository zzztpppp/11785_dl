import os
import torch
import torchvision

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "hw2p2_data")

# For the classfication task
training_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train_data"))
validation_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "val_data"))
test_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test_data"))

# For the verification task
verification_data = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "verification_data"))


# Construct model for the training process
class ClassificationNetwork(torch.nn.Module):
    pass
