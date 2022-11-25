import time
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.datasets import load_boston
from torch import nn

from FedProx_sim import (CentralizedModel, agent_trainer, co_aggregation,
                         data_splitter)
from utils import Logger, LogMode, LogType

import seaborn as sns; sns.set()


class InitRound:
    """
        This class is used by coordinator to send initial data to agents
    """
    def __init__(self, color: int, dataloader, n_epoch: int = 2,
        mu: float = 0.3, lr: float = 0.1, variable_E: bool = True) -> None:
        self.color = color
        self.dataloader = dataloader
        self.n_epoch = n_epoch
        self.mu = mu
        self.lr = lr
        self.variable_E = variable_E

class TrainRound:
    """
        This class is used by agents at the end of every training round to send updates to
        coordinator
    """
    def __init__(self, new_param: List[torch.Tensor], E: float) -> None:
        self.new_param = new_param
        self.E = E

class Protocol:
    """
        Base class for participants protocol stack
    """
    def __init__(self, comm: MPI.Intercomm, log_mode) -> None:
        self.comm = comm
        self.log_mode = log_mode
        self.rank = comm.Get_rank()
        self.loss_fn = nn.L1Loss()
        self.new_params_list = None
        self.new_param = None
        self.E = None
        self.model = None
        self.color = 1
    
    
    def gather_param(self, group, root: int):
        """
            This method gathers updated parameters from agents in every training round

            Parameters:
                1. group: Communication group
                2. root: Root node index collecting results
        """
        self.new_params_list = group.gather(TrainRound(self.new_param, self.E), root=root)
    
    def broadcast_model(self, group, root: int):
        """
            This method is used to broadcast new global model to agents
        """
        self.model = group.bcast(self.model, root=root)
        self.logger.log(f"Parameters after broadcast: {self.model}", LogType.debug)

class AgentProtocol(Protocol):
    """
        This class implements agents protocol stack
    """
    def __init__(self, comm: MPI.Intercomm, log_mode) -> None:
        super().__init__(comm, log_mode)
        self.logger = Logger("Agent", self.rank, log_mode)
    
    def respond_resource_req(self):
        """
            This method responds to coordinator resource request message
        """
        res = 1
        self.logger.log(f"Sending resource response, response = {res}", LogType.debug)
        self.comm.send(res, dest=0)
    
    def recv_color(self):
        """
            This method receives membership's flag for next training round and some initial data
            in "InitRound" objects
        """
        self.init_round: InitRound = self.comm.recv(source=0)
        self.logger.log(f"Membership status for this round: {self.init_round.color}", LogType.debug)    
    
    def update_params(self):
        """
            This method updates parameters locally
        """

        self.new_param, self.E = agent_trainer(
            model=self.model,
            n_epoch=self.init_round.n_epoch,
            train_loader=self.init_round.dataloader,
            lr=self.init_round.lr,
            loss_fn=self.loss_fn,
            mu=self.init_round.mu,
            variable_E=self.init_round.variable_E
        )


        

class COProtocol(Protocol):
    """
        This class implements coordinator protocol stack
    """
    def __init__(self, comm: MPI.Intercomm, log_mode, n_agents: int, logger_index: int, layer_size_vec) -> None:
        super().__init__(comm, log_mode)
        self.logger = Logger("CO", 0, log_mode)
        self.color = 1
        self.n_agents = n_agents
        self.test_loss_hist = []
        self.train_loss_hist = []
        self.round_hist = []
        self.logger_index = logger_index
        self.layer_size_vec = layer_size_vec
    
    def send_resource_req(self):
        """
            This method sends resource request to agents
        """

        res = {}
        for agent in range(1, self.comm.Get_size()):
            self.logger.log(f"Waiting for agent {agent} resource response", LogType.debug)
            res[agent] = self.comm.recv(source=agent)
        
        return res
    
    def agent_selector(self, selection_dict: dict) -> List:
        """
            This method selects agents according to client selection algorithm
        """

        color_list = list(selection_dict.keys())
        self.logger.log(f"Agents selected for this round: {color_list}", LogType.debug)
        return color_list
    
    def send_color(self, selection_dict: dict, loader, batch_size: int = 10,
            test_size: float = 0.2, non_iid: bool = True, n_epoch: int = 2,
            mu: float = 0.3, lr: float = 0.1, variable_E: bool = True):
        """
            This method sends membership flags for next training round and some initial values
            in form of "InitRound" objects
        """

        color_list = self.agent_selector(selection_dict)
        data_loader_list, (self.x_test, self.y_test, self.x_train, self.y_train) = data_splitter(
            loader=loader,
            batch_size=batch_size,
            n_agents=self.n_agents,
            test_size=test_size,
            random_size=non_iid
        )

        self.agents_weight = np.array([len(loader) for loader in data_loader_list])
        self.agents_weight = self.agents_weight/np.sum(self.agents_weight)

        for agent in range(1, self.comm.Get_size()):
            color = 1 if agent in color_list else 0
            init_round = InitRound(color, data_loader_list[agent-1], n_epoch, mu, lr, variable_E)
            self.comm.send(init_round, dest=agent)
    
    def initialize_params(self):
        """
            Parameters value initializer
        """
        self.model = CentralizedModel(self.layer_size_vec)
    
    def consolidate_params(self):
        """
            This method consolidates received parameters according to Federated Learning algorithm
        """
        self.new_params_list.pop(0)
        new_params = list(map(lambda x:x.new_param, self.new_params_list))
        self.logger.log(f"Received parameters: {self.model}", LogType.debug)
        self.model = co_aggregation(self.layer_size_vec, new_params, self.agents_weight)
        self.logger.log(f"New parameter value: {self.model}", LogType.debug)

    def train_round_stat_update(self, round_index: int):
        """
            This method is used by coordinator to update training statistics
        """
        self.E_hist_per_run = list(map(lambda x:x.E, self.new_params_list))

        with torch.no_grad():
            test_pred = self.model.forward(self.x_test)
            test_loss = self.loss_fn(test_pred, self.y_test)

            train_pred = self.model.forward(self.x_train)
            train_loss = self.loss_fn(train_pred, self.y_train)

            self.test_loss_hist.append(test_loss)
            self.train_loss_hist.append(train_loss)
            self.round_hist.append(round_index*self.logger_index)

class ResultCollector:
    """
        This class is used by coordinator to collect and generate final results
    """
    def __init__(self, n_round, n, n_agents) -> None:
        self.train_int_hist = np.zeros((n, n_round))
        self.test_int_hist = np.zeros((n, n_round))
        self.duration_hist = np.zeros(n)
        self.E_hist = np.zeros((n, n_agents))
    
    def update(self, run_index: int, train_loss_hist: List, test_loss_hist: List,
                duration: float, E_hist_per_run: List, round_hist: List):
        self.train_int_hist[run_index, :] = train_loss_hist
        self.test_int_hist[run_index, :] = test_loss_hist
        self.duration_hist[run_index] = duration
        self.E_hist[run_index, :] = E_hist_per_run
        self.round_hist = round_hist
    
    def plot_results(self, n: int, non_iid: bool, mu: float, variable_E: bool, lr: float):
        print(f"[INFO] Agents' average epoch: {np.mean(self.E_hist, axis=0)}")
    
        # CI calculation
        train_mean = np.mean(self.train_int_hist, axis=0)
        train_std = np.std(self.train_int_hist, axis=0)
        train_ci = 1.96*train_std/np.sqrt(n)

        test_mean = np.mean(self.test_int_hist, axis=0)
        test_std = np.std(self.test_int_hist, axis=0)
        test_ci = 1.96*test_std/np.sqrt(n)

        dur_mean = np.mean(self.duration_hist)
        dur_std = np.std(self.duration_hist)
        dur_ci = 1.96*dur_std/np.sqrt(n)

        # saving important results for future usage
        with open(f"FedProx_res_mu_{mu}_non_iid_{non_iid}_var_E_{variable_E}.npy", "wb") as file:
            np.save(file, test_mean)
            np.save(file, test_ci)

        print("[INFO] Plotting...")

        plt.plot(self.round_hist, train_mean, color="red")
        plt.fill_between(self.round_hist, train_mean-0.5*train_ci, train_mean+0.5*train_ci, color="red", alpha=0.3)
        plt.plot(self.round_hist, test_mean, color="blue")
        plt.fill_between(self.round_hist, test_mean-0.5*test_ci, test_mean+0.5*test_ci, color="blue", alpha=0.3)
        plt.legend(["train mean", "train 95% CI", "test mean", "test 95% CI"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"CI of train and test loss with n={n}, lr={lr}, $\mu$={mu} duration={dur_mean:0.2f} $\pm$ {dur_ci:0.2f}")
        plt.show()

def main(mu: float = 0.3, n_round: int = 20, batch_size: int = 1, test_size: float = 0.2,
        lr: float = 0.01, n_epoch: int = 2, logger_index: int = 1, n: int = 10, non_iid: bool = True,
        variable_E: bool = True, log_mode: LogMode = LogMode.debug):
    """
        Main

        parameters:
            1. n_round: Number of communication rounds
            2. log_mode: Log mode (OFF, INFO, DEBUG)
    """

    # initializing global variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_agents = comm.Get_size()-1
    layer_size_vec = [13, 64, 32, 1]

    if rank == 0:
        collector = ResultCollector(n_round, n, n_agents)
        print(f"""[INFO] parameters: batch_size: {batch_size}, test_size: {test_size}, lr: {lr}"""
        f""", n_epoch: {n_epoch}, logger_index: {logger_index}, n: {n}, mu: {mu}, non-iid: {non_iid}"""
        f""", variable_E: {variable_E}""")

    for run_index in range(n):
        if rank == 0:
            # Initializing coordinator protocol stack
            protocol = COProtocol(comm, log_mode, n_agents, logger_index, layer_size_vec)
            protocol.initialize_params()
            start_time = time.time()
        else:
            # Initializing agent protocol stack
            protocol = AgentProtocol(comm, log_mode)

        for round_index in range(n_round):
            if rank == 0:
                ### Coordinator code ###
                protocol.logger.log(f"Starting round {round_index}...", LogType.debug)
                
                # Step1: Sending resource request and receiving response from agents
                res = protocol.send_resource_req()

                # Step2: Announcing members of the next training round and sending their dataset
                protocol.send_color(
                    selection_dict=res,
                    loader=load_boston,
                    batch_size=batch_size,
                    test_size=test_size,
                    non_iid=non_iid,
                    n_epoch=n_epoch,
                    mu=mu,
                    lr=lr,
                    variable_E=variable_E
                )
            else:
                ### Agents code ###

                # Step1: Responding to resource request from coordinator
                protocol.respond_resource_req()

                # Step2: Receiving membership flags for next training round and some initial data
                protocol.recv_color()

            # Creating communication group for this round
            group = comm.Split(protocol.color, key=rank)

            # Step3: Starting training round by broadcasting global model
            protocol.broadcast_model(group, 0)

            # Code block only for members of new training round
            if protocol.color == 1:
                new_rank = group.Get_rank()
                protocol.logger.log(f"old rank: {rank}, new_rank: {new_rank}", LogType.debug)

                # Step3: Agents update model
                if new_rank != 0:
                    protocol.update_params()

                # Step4: Coordinator collecting updated models
                protocol.gather_param(group, 0)

                # Step4: Coordinator consolidating models
                if new_rank == 0:
                    protocol.consolidate_params()
                    protocol.train_round_stat_update(round_index)
        
        # coordinator collects statistics
        if rank == 0:
            collector.update(run_index, protocol.train_loss_hist, protocol.test_loss_hist,
                    time.time()-start_time, protocol.E_hist_per_run, protocol.round_hist)

    # coordinator generates final results
    if rank == 0:
        collector.plot_results(n, non_iid, mu, variable_E, lr)


if __name__ == "__main__":
    main(log_mode=LogMode.off)
