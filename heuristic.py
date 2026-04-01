import torch
import torch.nn as nn
from tm import *

class HeuristicModel:
    def __init__(self, width=128, layers=16):
        #Input: (write_symbol, next_state, move, read_symbol, current_state)
        model = nn.Sequential(nn.Linear(5, width))
        model.append(nn.ReLU())

        for _ in range(layers):
            model.append(nn.Linear(width, width))
            model.append(nn.ReLU())
        
        model.append(nn.Linear(width, 1))

    def action_heuristic(self, prior_state: TMState, trans: TMTransition) -> float:
        return self.model(get_action_tensor(prior_state, trans))