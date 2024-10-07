import numpy as np
import copy
import random


def apply_error_to_expert(expert, error_rate=0.1, choices=[-1, 0, 1]):
    new_expert = []
    for action in expert:
        if np.random.uniform() < error_rate:
            new_choices = [x for x in choices if x != action]
            new_choice = random.choice(new_choices)
            new_expert.append(new_choice)
    return np.array(new_expert)

def create_expert_table(base_expert, n_per_error=10, error_rates=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    experts = dict(base=base_expert)
    for error in error_rates:
        error_experts = []
        for _ in range(n_per_error):
            error_experts.append(apply_error_to_expert(base_expert, error_rate=error))
        experts[error] = np.array(error_experts)
    return experts



