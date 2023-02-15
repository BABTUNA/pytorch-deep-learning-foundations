# Dataset notes

A Dataset owns the indexing rule; a DataLoader owns batching, shuffling, and
worker processes. Keeping those responsibilities separate makes it possible to
reuse the same examples for training, validation, and inference.
