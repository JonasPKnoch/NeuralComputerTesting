from tm import *
import torch
import heapq as hq
from typing import List, Callable, Self

HeuristicFunction = Callable[[TMState, TMTransition], torch.tensor]

class TMSearchNode:
    def __init__(self, prior_state: TMState, transition: TMTransition, score: torch.tensor, parent: Self):
        self.prior_state = prior_state
        self.transition = transition
        self.score = score
        self.parent = parent
        self.best_descendant = self
        if parent:
            parent.update_descendant_score(self)

    def update_descendant_score(self, descendant: Self):
        if descendant.score < self.best_descendant.score:
            self.best_descendant = descendant
            if self.parent:
                self.parent.update_descendant_score(descendant)


    def __lt__(self, other):
        return self.score < other.score

def enumerate_search_nodes(heuristic: HeuristicFunction, prior_state: TMState, parent: TMSearchNode):
    for transition in prior_state.enumerate_transitions():
        yield TMSearchNode(prior_state, transition, heuristic(prior_state, transition), parent)

def tm_search(heuristic: HeuristicFunction, initial_nodes: List[TMSearchNode], expansions=128) -> List[TMSearchNode]:
    frontier = initial_nodes.copy()
    all_nodes = initial_nodes.copy()
    hq.heapify(frontier)

    for i in range(expansions):
        parent_scored_transition = hq.heappop(frontier)
        new_state = parent_scored_transition.prior_state.apply_transition(parent_scored_transition.transition)

        for child_scored_transition in enumerate_search_nodes(heuristic, new_state, parent_scored_transition):
            hq.heappush(frontier, child_scored_transition)
            all_nodes.append(child_scored_transition)
    
    return all_nodes