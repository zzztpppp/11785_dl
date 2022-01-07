import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import  ReduceLROnPlateau


MODE = "COMPLETE"
DATA = "FINAL"    # Whether we combine training and validation data together
device = torch.device("cuda")
current_context_size = 30


def get_data(mode):
    if mode != 'TOY':
        training_x = np.load("main_dataset/train.npy", allow_pickle=True)
        training_y = np.load("main_dataset/train_labels.npy", allow_pickle=True)

        val_x = np.load("main_dataset/dev.npy", allow_pickle=True)
        val_y = np.load("main_dataset/dev_labels.npy", allow_pickle=True)

        testing_x = np.load("main_dataset/test.npy", allow_pickle=True)
    else:
        training_x = np.load("toy_dataset/toy_train_data.npy", allow_pickle=True)
        training_y = np.load("toy_dataset/toy_train_label.npy", allow_pickle=True)

        val_x = np.load("toy_dataset/toy_val_data.npy", allow_pickle=True)
        val_y = np.load("toy_dataset/toy_val_label.npy", allow_pickle=True)

        testing_x = np.load("toy_dataset/toy_test_data.npy", allow_pickle=True)

    # Once we finish parameter tuning, we want to
    # get the most out of the data we have.
    if DATA == "FINAL":
        training_x = np.concatenate((training_x, val_x))
        training_y = np.concatenate((training_y, val_y))

    return training_x, training_y, val_x, val_y, testing_x


class HW1DataSet(torch.utils.data.Dataset):

    def __init__(self, x, y, context_size):

        self.x = x
        self.y = y
        index_map = []
        for i, x in enumerate(self.x):
            for j, _ in enumerate(x):
                index_pair = (i, j)
                index_map.append(index_pair)
        self.index_map = index_map
        self.context_size = context_size

        # Zero padding context
        for i, x in enumerate(self.x):
            self.x[i] = np.pad(x, ((self.context_size, self.context_size), (0, 0)), 'constant', constant_values=0)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        i, j = self.index_map[index]
        start_j = j
        end_j = start_j + 2 * self.context_size + 1
        x = self.x[i][start_j: end_j, :]
        y = self.y[i][start_j]
        return x.flatten(), y

    @staticmethod
    def collate_fn(batch):
        batch_x = np.stack([x for x, _ in batch])
        batch_y = np.stack([y for _, y in batch])
        batch_x = torch.tensor(batch_x).float()
        batch_y = torch.tensor(batch_y).long()

        return batch_x, batch_y


class HW1TestDataSet(torch.utils.data.Dataset):

    def __init__(self, x, context_size):
        self.x = x
        index_map = []
        for i, x in enumerate(self.x):
            for j, _ in enumerate(x):
                index_pair = (i, j)
                index_map.append(index_pair)
        self.index_map = index_map
        self.context_size = context_size
        # Zero padding context
        for i, x in enumerate(self.x):
            self.x[i] = np.pad(x, ((self.context_size, self.context_size), (0, 0)), 'constant', constant_values=0)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        i, j = self.index_map[index]
        start_j = j
        end_j = start_j + 2 * self.context_size + 1
        x = self.x[i][start_j: end_j, :]
        return x.flatten()

    @staticmethod
    def collate_fn(batch):
        batch_x = np.stack(batch)
        return torch.tensor(batch_x).float()


# Using a MLP training_model
class MLP(torch.nn.Module):
    def __init__(self, sizes, ndropout=0, nbatchnorm=0, dropout_p = 0.5):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            layers.append(torch.nn.Linear(in_size, out_size))
            if i != len(sizes):
                layers.append(torch.nn.ReLU())
            if i < nbatchnorm:
                layers.append(torch.nn.BatchNorm1d(out_size))
            if i < nbatchnorm + ndropout:
                layers.append(torch.nn.Dropout(p=dropout_p))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_epoch(data_loader, training_model, training_optimizer, loss_function):
    training_model.train()

    # For each batch
    train_loss_all = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        training_optimizer.zero_grad(set_to_none=True)
        out = training_model(batch_x)
        train_loss = loss_function(out, batch_y)
        train_loss.backward()
        training_optimizer.step()
        if i % 100000 == 0:
            print(train_loss.item())
        train_loss_all.append(train_loss.item())
    return np.mean(train_loss_all)


def evaluate_epoch(val_data_loader, eval_model, loss_func):
    predictions = []
    actuals = []

    eval_model.eval()
    for i, (inputs, targets) in enumerate(val_data_loader):
        # evaluate the training_model on the validation set
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            out = eval_model(inputs)
            # Calculate validation loss_function
            validation_loss = loss_func(out, targets)

        # retrieve numpy array
        out = out.cpu().detach().numpy()
        actual = targets.cpu().numpy()

        # convert to class labels
        out = np.argmax(out, axis=1)

        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        out = out.reshape((len(out), 1))
        # store
        predictions.append(out)
        actuals.append(actual)

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # Calculate validation accuracy
    acc = accuracy_score(actuals, predictions)
    return acc, validation_loss.item()


def predict(data_loader, pre_model):
    predictions = []
    pre_model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            out = pre_model(inputs)

            out = out.cpu().detach().numpy()
            out = np.argmax(out, axis=1)
            predictions.append(out)
    return predictions


if __name__ == "__main__":

    train_x, train_y, dev_x, dev_y, test_x = get_data(MODE)

    input_dimension = (1 + 2 * current_context_size) * 40
    output_dimension = 71
    size = [input_dimension, 4096, 4096, 4096, 4096, 4096, 4096, 4096, output_dimension]

    training_data = HW1DataSet(train_x, train_y, context_size=current_context_size)

    validation_data = HW1DataSet(dev_x, dev_y, context_size=current_context_size)

    test_data = HW1TestDataSet(test_x, context_size=current_context_size)

    data_loader_args = dict(shuffle=True, batch_size=8192, drop_last=True, collate_fn=HW1DataSet.collate_fn,
                            pin_memory=True, num_workers=8)

    training_data_loader = torch.utils.data.DataLoader(training_data, **data_loader_args)

    validation_data_loader = torch.utils.data.DataLoader(validation_data, shuffle=False, batch_size=4096,
                                                         drop_last=True, collate_fn=HW1DataSet.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=4096,
                                                   collate_fn=HW1TestDataSet.collate_fn)

    model = MLP(size, ndropout=7, dropout_p=0.5)
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=0, min_lr=0.001, factor=0.1)

    epochs = 90
    # We are going to adjust the learning rate every epoch
    # accroding the validation error.
    for epoch in range(epochs):
        # Train
        print(f"Current learning rate is {optimizer.param_groups[0]['lr']}")
        training_loss = train_epoch(training_data_loader, model, optimizer, loss)

        # Validation
        val_acc, val_loss = evaluate_epoch(validation_data_loader, model, loss)
        # scheduler.step(val_loss)
        # Print log of accuracy and loss_function
        print("Epoch: " + str(epoch) + ", Training loss_function: " + str(training_loss) + ", Validation loss_function:" + str(val_loss) +
              ", Validation accuracy:" + str(val_acc * 100) + "%")
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"model_{epoch}")

    # Predict labels in the test set
    ps = predict(test_data_loader, model)

    ps = np.concatenate(ps, axis=0).reshape((-1, 1))

    df_p = pd.DataFrame(data=ps, columns=['Label']).reset_index()

    df_p.columns = ['id', 'label']

    df_p.to_csv("hw1p2_submission.csv", index=False)

