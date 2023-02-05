import torch


def select_examples(values: torch.Tensor):
    first_row = values[0]
    last_column = values[:, -1]
    middle = values[1:-1, 1:-1]
    return first_row, last_column, middle
