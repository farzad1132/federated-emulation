from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

import seaborn as sns; sns.set()


class CustomDataSet(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, index) -> Tuple:
        return (self.x[index, :], self.y[index])


def create_data_loader(loader, batch_size: int, test_size: float = 0.2, transform: bool = True) \
    -> Tuple[DataLoader, Tuple]:
    """
        This function prepares data for training and testing process

        parameters:
            1. loader: sklearn dataset
            2. batch_size
            3. test_size: proportion of the whole dataset used for test
            4. transform: boolean indicating scaling of dataset
        
        returns:
            1. dataloader: training data loader
            2. x_test
            3. y_test
    """

    x, y = loader(return_X_y=True)
    if transform:
        x = scale(x)
        y = scale(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=32)
    train_dataset = CustomDataSet(x_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)

    return dataloader, (x_test, y_test)


class CentralizedModel(nn.Module):
    """
        MLP model used for modeling regression problem
    """

    def __init__(self, layer_size_vec: List) -> None:
        super().__init__()
        structure = []
        for i in range(len(layer_size_vec)-1):
            structure.append(nn.Linear(layer_size_vec[i], layer_size_vec[i+1]))
            if i != len(layer_size_vec)-2:
                structure.append(nn.ReLU())

        self.model = nn.Sequential(*structure)

    def forward(self, x):
        return self.model(x)


def trainer(model: nn.Module, n_epoch: int, train_loader: DataLoader, optimizer,
    loss_fn, x_test, y_test, logger_index: int = 1) -> Tuple:
    """
        This function trains a model

        parameters:
            1. model: pyTorch model
            2. n_epoch: number of epochs
            3. train_loader: training dataset loader
            4. optimizer: pyTorch optimizer
            5. loss_fn: loss function
            6. x_test: testing data points
            7. y_test: testing labels
            8. logger_index: used to indicate how to sample results from training process
        
        returns:
            1. epoch_list
            2. train_loss_hist: training loss history
            3. test_loss_hist: testing loss history 
    """

    y_test = torch.reshape(y_test, (-1, 1))
    y_train = torch.reshape(train_loader.dataset.y, (-1, 1))
    test_loss_hist = []
    train_loss_hist = []
    epoch_hist = []

    for epoch in range(n_epoch):
        for (x_b, y_b) in train_loader:
            y_b = torch.reshape(y_b, (-1, 1))
            optimizer.zero_grad()
            y_pred = model.forward(x_b)
            loss = loss_fn(y_pred, y_b)
            loss.backward()
            optimizer.step()

        if epoch % logger_index == 0:
            #print(f"[INFO] epoch: {epoch}/{n_epoch}")
            with torch.no_grad():
                test_pred = model.forward(x_test)
                test_loss = loss_fn(test_pred, y_test)

                train_pred = model.forward(train_loader.dataset.x)
                train_loss = loss_fn(train_pred, y_train)

                test_loss_hist.append(test_loss)
                train_loss_hist.append(train_loss)
                epoch_hist.append(epoch*logger_index)
    
    return (epoch_hist, train_loss_hist, test_loss_hist)


def main(batch_size: int = 10, test_size: float = 0.2, lr: float = 0.05,
        n_epoch: int = 60, logger_index: int = 1, transform: bool = True, n: int = 20):
    """
        Main function executing centralized learning procedure.

        parameters:
            1. batch_size
            2. test_size: proportion of the whole dataset used for test
            3. lr: learning rate
            4. n_epoch: number of epochs
            5. logger_index: used to indicate how to sample results from training process
            6. transform: boolean indicating scaling of dataset
            7. n: number of independent runs used for Confidence Interval calculation
    """
    
    train_int_hist = np.zeros((n, n_epoch))
    test_int_hist = np.zeros((n, n_epoch))

    print(f"""[INFO] parameters: batch_size: {batch_size}, test_size: {test_size}, lr: {lr}"""
        f""", n_epoch: {n_epoch}, logger_index: {logger_index}, n: {n}""")

    # loop over independent runs
    for run_index in range(n):

        train_data_loader, (x_test, y_test) = create_data_loader(
            loader=load_boston,
            batch_size=batch_size,
            test_size=test_size,
            transform=transform
        )

        # creating training pipeline
        model = CentralizedModel(layer_size_vec=(x_test.shape[1], 64, 32, 1))
        loss_fn = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=lr)

        #print("[INFO] Training start...")

        # training model
        epoch_hist, train_loss, test_loss = trainer(
            model=model,
            n_epoch=n_epoch,
            train_loader=train_data_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x_test=x_test,
            y_test=y_test,
            logger_index=logger_index
        )

        # saving results
        train_int_hist[run_index, :] = train_loss
        test_int_hist[run_index, :] = test_loss
    
    # CI calculation
    train_mean = np.mean(train_int_hist, axis=0)
    train_std = np.std(train_int_hist, axis=0)
    train_ci = 1.96*train_std/np.sqrt(n)

    test_mean = np.mean(test_int_hist, axis=0)
    test_std = np.std(test_int_hist, axis=0)
    test_ci = 1.96*test_std/np.sqrt(n)

    print("[INFO] Plotting...")

    plt.plot(epoch_hist, train_mean, color="red")
    plt.fill_between(epoch_hist, train_mean-0.5*train_ci, train_mean+0.5*train_ci, color="red", alpha=0.3)
    plt.plot(epoch_hist, test_mean, color="blue")
    plt.fill_between(epoch_hist, test_mean-0.5*test_ci, test_mean+0.5*test_ci, color="blue", alpha=0.3)
    plt.legend(["train mean", "train 95% CI", "test mean", "test 95% CI"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"CI of train and test loss with n={n}, lr={lr}")
    plt.show()

if __name__ == "__main__":
    main()