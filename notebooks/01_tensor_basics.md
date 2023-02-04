# Tensor basics

Scratch notes for creating tensors from lists and NumPy arrays. The exercises
compare shape, rank, dtype, and device, then reshape a flat batch into rows.

```python
x = torch.arange(12, dtype=torch.float32)
batch = x.reshape(3, 4)
```
