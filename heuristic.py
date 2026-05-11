import torch
import torch.nn as nn
from numpy import random
import tm
import mcts

class HeuristicModel:
    def __init__(self, tm_state: tm.TMState, width=128, layers=16):
        #Input: (read_symbol, current_state)
        self.model = nn.Sequential(nn.Linear(2, width))
        self.model.append(nn.ReLU())

        for _ in range(layers):
            self.model.append(nn.Linear(width, width))
            self.model.append(nn.ReLU())
        
        t_count = tm_state.transition_count()
        self.model.append(nn.Linear(width, t_count))
        self.model.append(nn.Softmax(dim=0))

    def action_heuristic(self, prior_state: tm.TMState) -> torch.Tensor:
        return self.model(tm.get_state_tensor(prior_state))
    
    def rollout_function(self, node: mcts.MCTSNode) -> float:
        current_state: tm.TMState = node.state
        transition_count = current_state.transition_count()

        while current_state.terminal_value() == None:
            next_transition_index = random.choice(transition_count, p=self.action_heuristic(current_state).detach().numpy())
            next_transition = current_state.get_transition_from_index(next_transition_index)
            current_state = current_state.apply_transition(next_transition)
        
        return current_state.terminal_value()