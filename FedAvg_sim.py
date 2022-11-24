import time
from copy import deepcopy
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
        self.y = torch.reshape(torch.as_tensor(y, dtype=torch.float32), (-1, 1))
        
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, index) -> Tuple:
        return (self.x[index, :], self.y[index])

    def __repr__(self) -> str:
        return f"Size={len(self)}"


def iid_data_splitter(loader, batch_size: int, n_agents: int, test_size: float = 0.2) \
    -> Tuple[Tuple[DataLoader], Tuple]:

    x, y = loader(return_X_y=True)
    x = scale(x)
    y = scale(y)
    y  = np.reshape(y, (-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True,
            random_state=np.random.randint(20, 40))

    per_agent = x_train.shape[0]//n_agents
    data_loader_list = []

    for i in range(n_agents):
        train_dataset = CustomDataSet(x_train[i*per_agent:(i+1)*per_agent, :], y_train[i*per_agent:(i+1)*per_agent])
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_loader_list.append(dataloader)

    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)
    y_test = torch.reshape(y_test, (-1, 1))

    return data_loader_list, (x_test, y_test, torch.as_tensor(x_train, dtype=torch.float32),
                        torch.as_tensor(y_train, dtype=torch.float32))


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

def parameter_parser(model: nn.Module, new: bool = True) -> List[torch.Tensor]:
    param_list = list(map(lambda x: x.data, list(model.parameters())))
    if new:
        return list(map(lambda x: x.detach().clone(), param_list))
    else:
        return param_list


def agent_trainer(model: nn.Module, n_epoch: int, train_loader: DataLoader, lr,
    loss_fn, mu: float, logger_index: int = 1) -> List[torch.Tensor]:

    def inexact_loss(mu: int, old_params: List[torch.Tensor], new_params: torch.Tensor) -> float:
        return 0.5*mu*np.sum([np.linalg.norm(n_par-o_par, 2)
            for (n_par, o_par) in zip(new_params, old_params)])
    
    local_model = deepcopy(model)
    optimizer = SGD(local_model.parameters(), lr=lr) 
    old_params = parameter_parser(local_model)
    for epoch in range(n_epoch):
        for (x_b, y_b) in train_loader:
            y_b = torch.reshape(y_b, (-1, 1))
            optimizer.zero_grad()
            y_pred = local_model.forward(x_b)
            loss = loss_fn(y_pred, y_b)
            loss += inexact_loss(mu, old_params, parameter_parser(local_model, False))

            loss.backward()
            optimizer.step()
    
    return parameter_parser(local_model)

def co_aggregation(layer_size_vec: List[int], new_params: List[List[torch.Tensor]],
                    weights: List[float]) -> nn.Module:

    new_model = CentralizedModel(layer_size_vec=layer_size_vec)
    n_agent = len(weights)

    for layer, layer_param in enumerate(new_model.parameters()):
        new_layer_par = torch.zeros(new_params[0][layer].size())

        for agent_index in range(n_agent):
           new_layer_par += new_params[agent_index][layer]*weights[agent_index]

        layer_param.data.add_(new_layer_par-layer_param.data)
    
    return new_model


def main(mu: float, n_agents: int = 3, n_round: int = 30, batch_size: int = 30, test_size: float = 0.2,
        lr: float = 0.05, n_epoch: int = 5, logger_index: int = 1, n: int = 20):
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
    
    train_int_hist = np.zeros((n, n_round))
    test_int_hist = np.zeros((n, n_round))
    duration_hist = np.zeros(n)

    print(f"""[INFO] parameters: batch_size: {batch_size}, test_size: {test_size}, lr: {lr}"""
        f""", n_epoch: {n_epoch}, logger_index: {logger_index}, n: {n}""")

    # loop over independent runs
    for run_index in range(n):

        test_loss_hist = []
        train_loss_hist = []
        round_hist = []

        data_loader_list, (x_test, y_test, x_train, y_train) = iid_data_splitter(
            loader=load_boston,
            batch_size=batch_size,
            n_agents=n_agents,
            test_size=test_size
        )
        

        # calculating agents weight
        agents_weight = np.array([len(loader) for loader in data_loader_list])
        agents_weight = agents_weight/np.sum(agents_weight)

        layer_size_vec=[x_test.shape[1], 64, 32, 1]

        # creating training pipeline
        loss_fn = nn.L1Loss()

        #print("[INFO] Training start...")
        global_model = CentralizedModel(layer_size_vec=layer_size_vec)

        start_time = time.time()

        for round_index in range(n_round):
            new_params = []
            for agent_index in range(n_agents):

                # agent model update
                # TODO: add random client selection
                new_params.append(
                    agent_trainer(
                        model=global_model,
                        n_epoch=n_epoch,
                        train_loader=data_loader_list[agent_index],
                        loss_fn=loss_fn,
                        mu=mu,
                        logger_index=logger_index,
                        lr=lr
                    )
                )

            global_model = co_aggregation(layer_size_vec, new_params, agents_weight)

            if round_index % logger_index == 0:
                #print(f"[INFO] epoch: {epoch}/{n_epoch}")
                with torch.no_grad():
                    test_pred = global_model.forward(x_test)
                    test_loss = loss_fn(test_pred, y_test)

                    train_pred = global_model.forward(x_train)
                    train_loss = loss_fn(train_pred, y_train)

                    test_loss_hist.append(test_loss)
                    train_loss_hist.append(train_loss)
                    round_hist.append(round_index*logger_index)

        # saving results
        train_int_hist[run_index, :] = train_loss_hist
        test_int_hist[run_index, :] = test_loss_hist
        duration_hist[run_index] = time.time() - start_time
    
    # CI calculation
    train_mean = np.mean(train_int_hist, axis=0)
    train_std = np.std(train_int_hist, axis=0)
    train_ci = 1.96*train_std/np.sqrt(n)

    test_mean = np.mean(test_int_hist, axis=0)
    test_std = np.std(test_int_hist, axis=0)
    test_ci = 1.96*test_std/np.sqrt(n)

    dur_mean = np.mean(duration_hist)
    dur_std = np.std(duration_hist)
    dur_ci = 1.96*dur_std/np.sqrt(n)

    print("[INFO] Plotting...")

    plt.plot(round_hist, train_mean, color="red")
    plt.fill_between(round_hist, train_mean-0.5*train_ci, train_mean+0.5*train_ci, color="red", alpha=0.3)
    plt.plot(round_hist, test_mean, color="blue")
    plt.fill_between(round_hist, test_mean-0.5*test_ci, test_mean+0.5*test_ci, color="blue", alpha=0.3)
    plt.legend(["train mean", "train 95% CI", "test mean", "test 95% CI"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"CI of train and test loss with n={n}, lr={lr}, duration:{dur_mean:0.2f} $\pm$ {dur_ci:0.2f}")
    plt.show()

if __name__ == "__main__":
    main(n=10, mu=0)