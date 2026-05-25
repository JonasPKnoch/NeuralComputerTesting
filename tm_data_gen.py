import torch
from torch.utils.data import Dataset
from tm import TuringMachine, TMDefinition, TMStatelessTransition, TMRuleSet
from typing import Iterable, Callable
from task_generators import TaskGenerator

def generate_tm_trace(initial_state: TuringMachine, max_depth = 10000) -> Iterable[TMStatelessTransition]:
    tail = initial_state
    yield tail.get_stateless_transition()
    while not tail.halting() and tail.depth < max_depth:
        tail = tail.transition()
        yield tail.get_stateless_transition()

class TMTransitionDataset(Dataset):
    def __init__(self, rules: TMRuleSet, definition: TMDefinition, task_generator: TaskGenerator, task_count=100):
        self.transition_list = []
        for _ in range(task_count):
            input, output = task_generator()
            initial_tm = TuringMachine(rules, definition, input)
            for read, (write, move) in generate_tm_trace(initial_tm):
                self.transition_list.append((torch.tensor([read]), torch.tensor([write, move])))
        
    def __len__(self):
        return len(self.transition_list)

    def __getitem__(self, index):
        return self.transition_list[index]