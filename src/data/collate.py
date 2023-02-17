import torch


def pad_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    width = int(lengths.max())
    padded = torch.zeros((len(sequences), width))
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
    return padded, torch.tensor(labels), lengths
