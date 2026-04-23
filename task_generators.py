import random
import torch
from typing import Tuple

def copy_task_generator(max_size=6) -> Tuple[torch.Tensor, torch.Tensor]:
    size = random.randint(1, max_size)
    input_vector = torch.full([size], 1, dtype=float)
    target_vector = torch.full([size*2 + 1], 1, dtype=float)
    target_vector[size] = 0

    
    return (input_vector, target_vector)