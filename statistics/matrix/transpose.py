# What is a Transpose Matrix?
# A transpose means:
# converting rows into columns
# and columns into rows
# Why Transpose is Important in ML

# Transpose is used everywhere in:

# neural networks
# linear regression
# deep learning
# covariance matrices
# gradient calculations
# embeddings
# similarity systems

# ML Intuition

# Transpose helps:

# align dimensions
# rotate vectors mathematically
# convert feature orientation
# # enable dot products

# What it does: Turns rows into columns ($A \rightarrow A^T$
# Shape Swapping: Flips dimensions cleanly from (M, N) to (N, M).
# Why ML needs it: To forcibly align inner dimensions so matrices can be multiplied without causing shape mismatch errors.

import numpy as np

A = np.array([[1, 2, 4],
              [5, 3, 2]])

print(f"\n print original matrix:\n {A}")

A_transpose = A.T

print(f"\n Transpose of matrix A:\n {A_transpose}")