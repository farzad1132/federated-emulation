from typing import List

import numpy as np
from mpi4py import MPI

from utils import Logger, LogMode, LogType


class Protocol:
    """
        Base class for participants protocol stack
    """
    def __init__(self, comm: MPI.Intercomm, log_mode) -> None:
        self.comm = comm
        self.log_mode = log_mode
        self.rank = comm.Get_rank()
        self.parameters = None
    
    def broadcast_param(self, group, root: int):
        """
            This method broadcasts parameters in a communication group

            Parameters:
                1. group: Communication group
                2. root: root node index that starts broadcast
        """
        self.parameters = group.bcast(self.parameters, root=root)
    
    def gather_param(self, group, root: int):
        """
            This method gathers updated parameters from members of a communication group

            Parameters:
                1. group: Communication group
                2. root: Root node index collecting results
        """
        self.parameters = group.gather(self.parameters, root=root)

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

        # TODO: Implement appropriate resource response method
        res = np.random.randint(0, 10)
        self.logger.log(f"Sending resource response, response = {res}", LogType.debug)
        self.comm.send(res, dest=0)
    
    def recv_color(self):
        """
            This method receives membership's flag for next training round
        """

        self.color = self.comm.recv(source=0)
        self.logger.log(f"Membership status for this round: {self.color}", LogType.debug)

    def broadcast_param(self, group, root: int):
        super().broadcast_param(group, root)
        self.logger.log(f"Parameters after broadcast: {self.parameters}", LogType.debug)
    
    def update_params(self):
        """
            This method updates parameters locally
        """

        # TODO: Implement
        self.parameters += 1

class COProtocol(Protocol):
    """
        This class implements coordinator protocol stack
    """
    def __init__(self, comm: MPI.Intercomm, log_mode) -> None:
        super().__init__(comm, log_mode)
        self.logger = Logger("CO", 0, log_mode)
        self.color = 1
    
    def send_resource_req(self):
        """
            This method sends resource request to agents
        """

        # TODO: Implement according to protocol
        res = {}
        for agent in range(1, self.comm.Get_size()):
            self.logger.log(f"Waiting for agent {agent} resource response", LogType.debug)
            res[agent] = self.comm.recv(source=agent)
        
        return res
    
    def agent_selector(self, resource_res: dict) -> List:
        """
            This method selects agents according to client selection algorithm
        """

        # TODO: Implement appropriate client selection algorithm
        color_list = list(resource_res.keys())
        self.logger.log(f"Agents selected for this round: {color_list}", LogType.debug)
        return color_list
    
    def send_color(self, color_list: List):
        """
            This method sends membership flags for next training round
        """

        for agent in range(1, self.comm.Get_size()):
            color = 1 if agent in color_list else 0
            self.comm.send(color, dest=agent)
    
    def initialize_params(self):
        """
            Parameters value initializer
        """

        # TODO: Implement appropriate initializing method
        self.parameters = 4
    
    def broadcast_param(self, group, root: int):
        super().broadcast_param(group, root)
        self.logger.log(f"Parameters after broadcast: {self.parameters}", LogType.debug)
    
    def consolidate_params(self):
        """
            This method consolidates received parameters according to Federated Learning algorithm
        """
        
        # TODO: Implement
        self.parameters.pop(0)
        self.logger.log(f"Received parameters: {self.parameters}", LogType.debug)
        self.parameters = sum(self.parameters)
        self.logger.log(f"New parameter value: {self.parameters}", LogType.debug)

def main(n_round: int, log_mode: LogMode = LogMode.debug):
    """
        Main

        parameters:
            1. n_round: Number of communication rounds
            2. log_mode: Log mode (OFF, INFO, DEBUG)
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Initializing coordinator protocol stack
        protocol = COProtocol(comm, log_mode)
        protocol.initialize_params()
    else:
        # Initializing agent protocol stack
        protocol = AgentProtocol(comm, log_mode)

    for round in range(n_round):
        if rank == 0:
            ### Coordinator code ###
            protocol.logger.log(f"Starting round {round}...", LogType.info)
            
            # Step1: Sending resource request and receiving response from agents
            res = protocol.send_resource_req()

            # Step2: Selecting agents for next training round
            color_list = protocol.agent_selector(res)

            # Step2: Announcing members of the next training round
            protocol.send_color(color_list)
        else:
            ### Agents code ###

            # Step1: Responding to resource request from coordinator
            protocol.respond_resource_req()

            # Step2: Receiving membership flags for next training round
            protocol.recv_color()

        # Creating communication group for this round
        group = comm.Split(protocol.color, key=rank)

        # Step3: Starting training round by broadcasting parameters
        protocol.broadcast_param(group, 0)

        # Code block only for members of new training round
        if protocol.color == 1:
            new_rank = group.Get_rank()
            protocol.logger.log(f"old rank: {rank}, new_rank: {new_rank}", LogType.debug)

            # Step3: Agents update parameters
            if new_rank != 0:
                protocol.update_params()

            # Step4: Coordinator collecting updated parameters
            protocol.gather_param(group, 0)

            # Step4: Coordinator consolidating parameters
            if new_rank == 0:
                protocol.consolidate_params()


if __name__ == "__main__":
    main(n_round=2)