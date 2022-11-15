from typing import List, Tuple

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

    loss_hist = []
    train_loss_hist = []
    epoch_hist = []

    for epoch in range(n_epoch):
        train_loss = 0
        for (x_b, y_b) in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x_b)
            loss = loss_fn(y_pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % logger_index == 0:
            print(f"[INFO] epoch: {epoch}/{n_epoch}")
            with torch.no_grad():
                test_pred = model.forward(x_test)
                test_loss = loss_fn(test_pred, y_test)

                loss_hist.append(test_loss)
                train_loss_hist.append(train_loss)
                epoch_hist.append(epoch)
    
    return (epoch_hist, train_loss_hist, loss_hist)


def main(batch_size: int = 10, test_size: float = 0.2, lr: float = 0.001,
        n_epoch: int = 30, logger_index: int = 1, transform: bool = True):
    
    print(f"""[INFO] parameters: batch_size: {batch_size}, test_size: {test_size}, lr: {lr}"""
        f""", n_epoch: {n_epoch}, logger_index: {logger_index}""")

    train_data_loader, (x_test, y_test) = create_data_loader(
        loader=load_boston,
        batch_size=batch_size,
        test_size=test_size,
        transform=transform
    )

    model = CentralizedModel(layer_size_vec=(x_test.shape[1], 64, 32, 1))
    loss_fn = nn.L1Loss()
    optimizer = SGD(model.parameters(), lr=lr)

    print("[INFO] Training start...")

    epoch_hist, train_loss, loss_hist = trainer(
        model=model,
        n_epoch=n_epoch,
        train_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        x_test=x_test,
        y_test=y_test,
        logger_index=logger_index
    )

    print("[INFO] Plotting...")

    plt.plot(epoch_hist, train_loss, color="black")
    plt.plot(epoch_hist, loss_hist, color="blue")
    plt.legend(["train", "test"])
    plt.show()

if __name__ == "__main__":
    main()