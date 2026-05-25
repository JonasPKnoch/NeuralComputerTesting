import torch
from typing import List, Dict, Tuple, Self

class TMDefinition:
    def __init__(self, symbol_count: int, state_count: int, move_count: int, halt_state:int=1):
        self.symbol_count = symbol_count
        self.state_count = state_count
        self.move_count = move_count
        self.halt_state = halt_state
    
    def rule_valid(self, head: Tuple[int, int], body: Tuple[int, int, int]) -> bool:
        state, read = head
        write, move, new_state = body
        return \
            state != self.halt_state and state in range(self.state_count) and\
            read in range(self.symbol_count) and\
            write in range(self.symbol_count) and\
            new_state in range(self.state_count) and\
            move in range(-self.move_count, self.move_count) and move != 0

TMStatelessTransition = Tuple[int, Tuple[int, int]]
TMRuleSet = Dict[Tuple[int, int], Tuple[int, int, int]]

class TuringMachine:
    def __init__(self, rules: TMRuleSet,
                 definition: TMDefinition,
                 tape: List[int] = [], position: int = 0, state: int = 1, 
                 previous_state: Self = None):
        self.rules = rules
        self.definition = definition
        for head, body in rules:
            assert definition.rule_valid(head, body)
        
        self.tape = tape
        self.position = position
        self.state = state
        self.previous_state = previous_state
        if previous_state == None:
            self.depth = 1
        else:
            self.depth = previous_state.depth + 1
    
    def halting(self) -> bool:
        return self.state == self.definition.halt_state
    
    def transition(self) -> Self:
        if self.halting():
            return None
        
        write, move, new_state = self.rules[self.get_rule_head()]

        new_tape = self.tape.copy()
        new_tape[self.position] = write

        new_position = self.position + move
        if new_position > len(new_tape):
            new_tape.extend([0 for _ in range(len(new_tape))])
        if new_position < 0:
            new_position = 0
        
        return TuringMachine(self.rules, new_tape, new_position, new_state, self)
        
    def get_rule_head(self) -> Tuple[int, int]:
        return (self.state, self.tape[self.position])
    
    def get_stateless_transition(self) -> TMStatelessTransition:
        rule_head = self.get_rule_head()
        write, move, _ = self.rules[rule_head]
        _, read = rule_head

        return (read, (write, move))