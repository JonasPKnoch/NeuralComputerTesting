from tm import *
from tm_search import *
from search_heuristics import *
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

def get_search_heuristic_training_samples(heuristic, input_memory, expansions=64):
    state = TMState(input_memory)
    initial_nodes = list(enumerate_search_nodes(heuristic, state, None))

    all_nodes = tm_search(heuristic, initial_nodes, expansions=expansions)

    for node in all_nodes:
        yield (get_action_tensor(node.prior_state, node.transition), node.score)

def get_search_heuristic_training_samples_from_task(heuristic, task_generator, task_count=16, expansions=64):
    for _ in range(task_count):
        input_memory, target_memory = task_generator()
        for sample in get_search_heuristic_training_samples(heuristic, input_memory, expansions):
            yield sample

def get_distance_heuristic_training_samples_from_task(task_generator, task_count=16, expansions=64):
    for _ in range(task_count):
        input_memory, target_memory = task_generator()
        heuristic = get_distance_heuristic(target_memory)
        for sample in get_search_heuristic_training_samples(heuristic, input_memory, expansions):
            yield sample

def get_training_batches(training_samples, batch_size=64):
    actions_batch = torch.zeros([batch_size, 5])
    scores_batch = torch.zeros([batch_size, 1])
    batch_index = 0

    for action, score in training_samples:
        actions_batch[batch_index] = action
        scores_batch[batch_index] = score
        batch_index += 1
        if batch_index == batch_size:
            yield actions_batch, scores_batch
            batch_index = 0


def train_heuristic(heuristic_model, training_data, epochs = 64):
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(heuristic_model.parameters())
    start_time = time.time()

    for i in range(epochs):
        running_loss = 0
        training_samples = 0

        for actions, values in training_data:
            output = heuristic_model(actions)
            loss = loss_function(output, values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            training_samples += 1
        
        print(f"Epoch {i}/{epochs} - Loss {running_loss/training_samples} - Time {time.time() - start_time}s")