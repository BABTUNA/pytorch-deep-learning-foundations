# Keeping track of shapes

Most tensor mistakes in these exercises came from losing the batch dimension.
Write shapes beside each step and prefer `reshape` over guessing dimensions.

- tabular batch: `(examples, features)`
- image batch: `(examples, channels, height, width)`
- class scores: `(examples, classes)`
