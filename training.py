import mcts
import tm
import heuristic
import torch
from torch import nn
from typing import List, Callable, Tuple
import random

# Select training samples randomly from previous many MCTS runs

def generate_training_samples(
        heuristic_model: heuristic.HeuristicModel,
        task_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]], 
        games = 10, 
        iterations = 10000,
        shuffle=True
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    samples = []

    for i in range(games):
        input_memory, target_memory = task_generator()
        tm_state = tm.TMState(input_memory, target_memory)
        print(f"Generating data for target: {target_memory}")

        node_path = mcts.play_mcts_game(mcts.MCTSNode(tm_state), heuristic_model, iterations)

        for j in range(len(node_path) - 1):
            prior_state = node_path[j].state
            next_state = node_path[j + 1].state
            transition_index = next_state.get_index_from_transition(next_state.last_transition)

            sample = (tm.get_state_tensor(prior_state), nn.functional.one_hot(torch.tensor(transition_index, dtype=torch.long), next_state.transition_count()).detach())
            samples.append(sample)
        
        print(f"Generated samples for game {i}/{games}")

    if shuffle:
        random.shuffle(samples)

    return samples

def save_training_data(training_data: List, path: str):
    torch.save(training_data, f"./training_data/{path}")
