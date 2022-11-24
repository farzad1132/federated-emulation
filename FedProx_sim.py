import math
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


def data_splitter(loader, batch_size: int, n_agents: int, test_size: float = 0.2,
                random_size: bool = False) -> Tuple[Tuple[DataLoader], Tuple]:
    """
        This function is used for dividing data between agents. Note that this method
        is capable of generating iid and non-iid datasets using "random_size" parameter.
    """

    def gen_random_size(n_agents: int, size: int) -> List[Tuple]:
        """
            This function generates random sizes for datasets in non-iid setting
        """
        done = False
        while not done:
            rand_digit = list(np.random.randint(1, size, size=n_agents-1))
            rand_digit.insert(0, 0)
            rand_digit.append(size)

            done = True
            for index in range(len(rand_digit)-1):
                if rand_digit[index+1]-rand_digit[index] <= size/10:
                    done = False
                    break
        
        index_list = [(rand_digit[i], rand_digit[i+1]) for i in range(len(rand_digit)-1)]
        return index_list
            

    x, y = loader(return_X_y=True)
    x = scale(x)
    y = scale(y)
    y  = np.reshape(y, (-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True,
                                                    random_state=np.random.randint(20, 40))

    per_agent = x_train.shape[0]//n_agents
    data_loader_list = []

    if random_size:
        random_index = gen_random_size(n_agents, x_train.shape[0])

    for i in range(n_agents):
        if not random_size:
            train_dataset = CustomDataSet(x_train[i*per_agent:(i+1)*per_agent, :],
                                    y_train[i*per_agent:(i+1)*per_agent])
        else:
            train_dataset = CustomDataSet(x_train[random_index[i][0]:random_index[i][1], :],
                                    y_train[random_index[i][0]:random_index[i][1]])

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_loader_list.append(dataloader)

    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)
    y_test = torch.reshape(y_test, (-1, 1))

    return data_loader_list, (x_test, y_test, torch.as_tensor(x_train, dtype=torch.float32),
                        torch.as_tensor(y_train, dtype=torch.float32))


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
    """
        This function parses pyTorch's model to get into its parameters weights which
        is in tensor format
    """
    param_list = list(map(lambda x: x.data, list(model.parameters())))
    if new:
        return list(map(lambda x: x.detach().clone(), param_list))
    else:
        return param_list


def agent_trainer(model: nn.Module, n_epoch: int, train_loader: DataLoader, lr,
    loss_fn, mu: float, variable_E: bool = True) -> Tuple[List[torch.Tensor], int]:
    """
        Agent's trainer function
    """

    def inexact_loss(mu: int, old_params: List[torch.Tensor], new_params: torch.Tensor) -> float:
        """
            proximity loss (used for FedProx)
        """
        return 0.5*mu*np.sum([np.linalg.norm(n_par-o_par, 2)
            for (n_par, o_par) in zip(new_params, old_params)])
    
    local_model = deepcopy(model)
    optimizer = SGD(local_model.parameters(), lr=lr) 
    old_params = parameter_parser(local_model)

    # If "variable_E" is true then variable number of epochs will be selected for each agent
    # depending on their dataset size
    if variable_E:
        n_epoch = math.ceil(len(train_loader)/15)
    for _ in range(n_epoch):
        for (x_b, y_b) in train_loader:
            y_b = torch.reshape(y_b, (-1, 1))
            optimizer.zero_grad()
            y_pred = local_model.forward(x_b)
            loss = loss_fn(y_pred, y_b)
            loss += inexact_loss(mu, old_params, parameter_parser(local_model, False))

            loss.backward()
            optimizer.step()
    
    return parameter_parser(local_model), n_epoch

def co_aggregation(layer_size_vec: List[int], new_params: List[List[torch.Tensor]],
                    weights: List[float]) -> nn.Module:
    """
        This function is used by aggregator to average local models
    """

    new_model = CentralizedModel(layer_size_vec=layer_size_vec)
    n_agent = len(weights)

    for layer, layer_param in enumerate(new_model.parameters()):
        new_layer_par = torch.zeros(new_params[0][layer].size())

        for agent_index in range(n_agent):
           new_layer_par += new_params[agent_index][layer]*weights[agent_index]

        layer_param.data.add_(new_layer_par-layer_param.data)
    
    return new_model


def main(mu: float = 0, n_agents: int = 3, n_round: int = 30, batch_size: int = 10,
        test_size: float = 0.2, lr: float = 0.05, n_epoch: int = 2,
        logger_index: int = 1, n: int = 10, non_iid: bool = True, variable_E: bool = False):
    """
        Main function executing centralized learning procedure.

        some parameters:
            mu: FedProx parameter
            n: number of independent runs used for Confidence Interval calculation
            variable_E: If this is True, then each agent will train their models within variable amount
                        epochs depending on their dataset size
    """
    
    # initializing some history tracker values
    train_int_hist = np.zeros((n, n_round))
    test_int_hist = np.zeros((n, n_round))
    duration_hist = np.zeros(n)
    E_hist = np.zeros((n, n_agents))

    print(f"""[INFO] parameters: batch_size: {batch_size}, test_size: {test_size}, lr: {lr}"""
        f""", n_epoch: {n_epoch}, logger_index: {logger_index}, n: {n}, mu: {mu}, non-iid: {non_iid}"""
        f""", variable_E: {variable_E}""")

    # loop over independent runs
    for run_index in range(n):

        test_loss_hist = []
        train_loss_hist = []
        round_hist = []

        # attaining agents' dataset
        data_loader_list, (x_test, y_test, x_train, y_train) = data_splitter(
            loader=load_boston,
            batch_size=batch_size,
            n_agents=n_agents,
            test_size=test_size,
            random_size=non_iid
        )
        

        # calculating agents weight
        agents_weight = np.array([len(loader) for loader in data_loader_list])
        agents_weight = agents_weight/np.sum(agents_weight)

        # model creation
        layer_size_vec=[x_test.shape[1], 64, 32, 1]
        global_model = CentralizedModel(layer_size_vec=layer_size_vec)

        # creating training pipeline
        loss_fn = nn.L1Loss()

        # timer start
        start_time = time.time()

        # start training
        for round_index in range(n_round):
            new_params = []
            E_hist_per_run = []

            for agent_index in range(n_agents):
                # agent model update
                param, E = agent_trainer(
                                model=global_model,
                                n_epoch=n_epoch,
                                train_loader=data_loader_list[agent_index],
                                loss_fn=loss_fn,
                                mu=mu,
                                lr=lr,
                                variable_E=variable_E
                            )
                # getting agents training results
                new_params.append(param)
                E_hist_per_run.append(E)

            # creating new global model
            global_model = co_aggregation(layer_size_vec, new_params, agents_weight)

            # updating statistics
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
        E_hist[run_index, :] = E_hist_per_run
    
    print(f"[INFO] Agents' average epoch: {np.mean(E_hist, axis=0)}")
    
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

    # saving important results for future usage
    with open(f"FedProx_res_mu_{mu}_non_iid_{non_iid}_var_E_{variable_E}.npy", "wb") as file:
        np.save(file, test_mean)
        np.save(file, test_ci)

    print("[INFO] Plotting...")

    plt.plot(round_hist, train_mean, color="red")
    plt.fill_between(round_hist, train_mean-0.5*train_ci, train_mean+0.5*train_ci, color="red", alpha=0.3)
    plt.plot(round_hist, test_mean, color="blue")
    plt.fill_between(round_hist, test_mean-0.5*test_ci, test_mean+0.5*test_ci, color="blue", alpha=0.3)
    plt.legend(["train mean", "train 95% CI", "test mean", "test 95% CI"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"CI of train and test loss with n={n}, lr={lr}, $\mu$={mu} duration={dur_mean:0.2f} $\pm$ {dur_ci:0.2f}")
    plt.show()

if __name__ == "__main__":
    main()
