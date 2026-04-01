import math
import random
from typing import Iterable, Callable, Self, Optional, List

class MCTSState:
    def all_next_states(self) -> Iterable[Self]:
        raise NotImplementedError("This method needs to be overwritten")
    
    def random_next_state(self) -> Self:
        next_states = list(self.all_next_states())
        return next_states[random.randint(0, len(next_states) - 1)]
    
    def terminal_value(self) -> Optional[float]:
        raise NotImplementedError("This method needs to be overwritten")

class MCTSNode:
    def __init__(self, state: MCTSState, parent: Self = None):
        self.state = state
        self.total_score = 0
        self.number_visits = 0
        self.parent = parent
        self.children = []

    def add_children(self):
        self.children = [MCTSNode(child, self) for child in self.state.all_next_states()]

class MCTS:
    def __init__(self, root_node: MCTSNode, rollout_function: Callable[[MCTSNode], float]):
        self.root = root_node
        self.rollout = rollout_function

    def perform_iteration(self):
        current = self.select_current_node()
        
        if current.number_visits > 0:
            current.add_children()
            current = current.children[0]
        
        score = self.rollout(current)

        self.backpropagate_values(current, score)
    
    def backpropagate_values(self, current: MCTSNode, score: float):
        current.total_score += score
        current.number_visits += 1
        if current.parent != None:
            self.backpropagate_values(current.parent, score)

    
    def select_current_node(self) -> MCTSNode:
        current = self.root

        while current.children:
            best_child = None
            best_ucb1 = -1
            for child in current.children:
                child_ucb1 = self.ucb1(child)
                if child_ucb1 > best_ucb1:
                    best_child = child
                    best_ucb1 = child_ucb1
            current = best_child

        return current

    def ucb1(self, node: MCTSNode):
        if node.number_visits == 0.0:
            return 999999.0
        n_inverse = 1.0/node.number_visits

        exploit_term = node.total_score*n_inverse
        explore_term = math.sqrt(math.log(node.parent.number_visits)*n_inverse)

        return exploit_term + 2.0*explore_term

def play_mcts_game(root_node: MCTSNode, rollout_function: Callable[[MCTSNode], float], iterations = 10000) -> List[List[MCTSNode]]:
    node_path = []
    current_root = root_node

    while current_root.state.terminal_value() != None:
        mcts = MCTS(current_root, rollout_function)
        for i in range(iterations):
            mcts.perform_iteration()
        node_path.append(current_root.children)
        best_child = current_root.children[0]
        for child in current_root.children[1:]:
            if child.number_visits > best_child.number_visits:
                best_child = child
        current_root = best_child
            


def random_rollout(node: MCTSNode) -> float:
    current_state = node.state

    while current_state.terminal_value() == None:
        current_state = current_state.random_next_state()
    
    return current_state.terminal_value()
        
