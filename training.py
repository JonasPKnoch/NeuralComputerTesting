from mcts import *
from tm import *
from heuristic import *
import torch
import torch.nn as torch
from typing import List, Callable, Tuple

# Select training samples randomly from previous many MCTS runs

def generate_training_samples(
        task_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]], 
        samples = 1000, 
        rollout_function: Callable[[MCTSNode], float] = random_rollout
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    
    input_memory, target_memory = task_generator()
    tm = TMState(input_memory, target_memory)

    mcts = MCTS(MCTSNode(tm), rollout_function)
