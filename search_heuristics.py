from tm import *
import torch
import torch.nn as nn
import random

class MLPHeuristic:
    def __init__(self, width=128, layers=16):
        #Input: (write_symbol, next_state, move, read_symbol, current_state)
        self.model = nn.Sequential(nn.Linear(5, width))
        self.model.append(nn.ReLU())

        for _ in range(layers):
            self.model.append(nn.Linear(width, width))
            self.model.append(nn.ReLU())
        
        self.model.append(nn.Linear(width, 1))

    def __call__(self, state: TMState, transition: TMTransition) -> torch.tensor:
        tensor_input = get_action_tensor(state, transition)
        
        return self.model(tensor_input)

def get_distance_heuristic(target_memory, state_cost=0.1):
    distance_function = nn.L1Loss(reduction = 'sum')

    def heuristic(state: TMState, transition: TMTransition) -> torch.tensor:
        nonlocal target_memory
        nonlocal distance_function

        new_memory = state.memory.detach()
        new_memory[state.position] = transition.write_symbol
        
        memory_size = new_memory.shape[0]
        target_size = target_memory.shape[0]
        if memory_size < target_size:
                new_memory = nn.functional.pad(new_memory, (0, target_size - memory_size))
        elif memory_size > target_size:
            target_memory = nn.functional.pad(target_memory, (0, memory_size - target_size))

        return distance_function(new_memory, target_memory) + (transition.new_state/state.state_count)*state_cost #This just ensures that some states are preferable to others, which will discourange random states

    return heuristic