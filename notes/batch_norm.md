# Batch normalization

Batch normalization keeps running statistics during training and uses them
during evaluation. This is one reason `model.train()` and `model.eval()` matter
even when gradient tracking is handled separately.
