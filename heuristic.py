import torch
import torch.nn as nn
from numpy import random
from tm import *
from mcts import *

class HeuristicModel:
    def __init__(self, tm: TMState, width=128, layers=16):
        #Input: (read_symbol, current_state)
        model = nn.Sequential(nn.Linear(2, width))
        model.append(nn.ReLU())

        for _ in range(layers):
            model.append(nn.Linear(width, width))
            model.append(nn.ReLU())
        
        model.append(nn.Linear(width, tm.transition_count))
        model.append(nn.Softmax())

    def action_heuristic(self, prior_state: TMState) -> torch.Tensor:
        return self.model(get_state_tensor(prior_state))
    
    def rollout_function(self, node: MCTSNode) -> float:
        current_state: TMState = node.state
        transition_count = current_state.transition_count()

        while current_state.terminal_value() == None:
            next_transition_index = random.choice(transition_count, p=self.action_heuristic(current_state))
            next_transition = current_state.get_transition_from_index(next_transition_index)
            current_state = current_state.apply_transition(next_transition)
        
        return current_state.terminal_value()