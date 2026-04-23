import mcts
import torch
import torch.nn as nn
from typing import Self, Iterable, Optional
import random

class TMTransition:
    def __init__(self, write_symbol: int, new_state: int, move: int):
        self.write_symbol = write_symbol
        self.new_state = new_state
        self.move = move

class TMState(mcts.MCTSState):
    def __init__(self, initial_memory: torch.tensor, target_memory: torch.tensor, initial_position=0, initial_state=0, symbol_count=2, state_count=8, move_count=4):
        self.symbol_count = symbol_count
        self.state_count = state_count
        self.move_count = move_count
        
        self.memory = initial_memory
        self.position = initial_position
        self.state = initial_state
        self.target_memory = target_memory
        self.target_size = target_memory.shape[0]
        self.inverse_target_size = 1.0/self.target_size

        self.last_frame = None
        self.last_transition = None
        self.run_steps = 0

        self.distance_function = nn.L1Loss()

    def all_next_states(self) -> Iterable[Self]:
        for transition in self.enumerate_transitions():
            yield self.apply_transition(transition)
    
    def random_next_state(self) -> Self:
        transition = TMTransition(random.randint(0, self.symbol_count-1), random.randint(0, self.state_count-1), random.randint(1, self.move_count)*random.choice([-1, 1]))
        return self.apply_transition(transition)
    
    def terminal_value(self) -> Optional[float]:
        if self.state != self.state_count - 1:
            return None

        fit_memory = self.memory
        fit_target = self.target_memory
        memory_size = self.memory.shape[0]
        if memory_size < self.target_size:
            fit_memory = nn.functional.pad(fit_memory, (0, self.target_size - memory_size))
        elif memory_size > self.target_size:
            fit_target = nn.functional.pad(fit_target, (0, memory_size - self.target_size))
        
        closeness = 1.0 - self.distance_function(fit_memory, fit_target)
        step = self.inverse_target_size/self.run_steps

        return closeness*2.0 + step
    
    def get_read_symbol(self) -> float:
        return self.memory[self.position]
    
    def apply_transition(self, transition: TMTransition) -> Self:
        assert transition.write_symbol in range(0, self.symbol_count)
        assert transition.new_state in range(0, self.state_count)
        assert transition.move in range(-self.move_count - 1, self.move_count + 1) and transition.move != 0

        new_state = transition.new_state
        new_position = self.position + transition.move
        if new_position < 0:
            new_position = 0

        mem_size = self.memory.shape[0]
        padding_right = 0
        while new_position >= mem_size:
            padding_right += mem_size
            mem_size += mem_size
        new_memory = nn.functional.pad(self.memory, (0, padding_right))
        new_memory[self.position] = transition.write_symbol

        new_tm = TMState(new_memory, self.target_memory, new_position, new_state, self.symbol_count, self.state_count, self.move_count)
        new_tm.last_frame = self
        new_tm.last_transition = transition
        new_tm.run_steps = self.run_steps + 1

        return new_tm

    def enumerate_transitions(self):
        for symbol in range(self.symbol_count):
            for state in range(self.state_count):
                for move in range(1, self.move_count + 1):
                    for sign in [-1, 1]:
                        yield TMTransition(symbol, state, move*sign)
    
    def get_transition_from_index(self, index: int) -> TMTransition:
        # Equivalent to list(enumerate_transitions())[index]
        sign = [-1, 1][index%2]
        index -= index%2
        
        move = index%self.move_count
        index -= index%self.move_count

        state = index%self.state_count
        index -= index%self.state_count
        
        symbol = index%self.symbol_count
        index -= index%self.symbol_count

        if index != 0:
            raise Exception("Something ain't right")
        
        return TMTransition(symbol, state, move*sign)



    def transition_count(self):
        return self.symbol_count*self.state_count*self.move_count*2
    
    def __str__(self):
        result = f"POS:{self.position}, STATE:{self.state}, MEM:"
        for i in range(self.memory.shape[0]):
            result += " " + str(int(self.memory[i]))
        
        return result

def get_action_tensor(prior_state: TMState, transition: TMTransition):
    return torch.tensor(
            [transition.write_symbol, transition.new_state, transition.move, prior_state.get_read_symbol(), prior_state.state])

def get_state_tensor(prior_state: TMState):
    return torch.tensor([prior_state.get_read_symbol(), prior_state.state])

