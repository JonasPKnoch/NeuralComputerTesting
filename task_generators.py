import random
import torch
from typing import Tuple, List, Callable

TaskGenerator = Callable[[], Tuple[List[int], List[int]]]

def copy_task_generator(max_size=4) -> Tuple[List[int], List[int]]:
    size = random.randint(1, max_size)
    input_vector = [1]*size
    target_vector = [1]*(size*2 + 1)
    target_vector[size] = 0
    
    return (input_vector, target_vector)