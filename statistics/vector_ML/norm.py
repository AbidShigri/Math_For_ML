# In Machine Learning, vector normalization means:
# Changing the size (magnitude/length) of a vector while keeping its direction the same.
# It helps make data values comparable and stable for ML models.
# Main Idea of Normalization
# Normalization usually converts a vector into a unit vector.
# A unit vector has magnitude (length) = 1.

import numpy as np

# original vector = v
v = np.array([1, 4, 3])

# magnitude = |v|

# np.linalg.norm()     # magnitude
# np.linalg.inv()      # matrix inverse
# np.linalg.det()      # determinant


magnitude = np.linalg.norm(v) #linalg means linear algebra it is a module inside a numpy

# normalize vector v^

normalized_v = v / magnitude

print(f"original vector {v}")
print(f"magnitude of orignal vector {magnitude}")
print(f"normalzied vector {normalized_v}")


# What is np.linalg ?

# linalg means:

# Linear Algebra

# It is a module inside NumPy.

# Linear Algebra is the mathematics of:

# vectors
# matrices
# tensors
# equations
# transformations

# These are the foundation of Machine Learning.

# What Does np.linalg Contain?

# It contains functions for:

# vector magnitude
# matrix multiplication
# determinant
# inverse
# eigenvalues
# solving equations



import numpy as np 
import torch

# simplest weight or error vector
w = [5.0, -3.1]
# numpy implementation to find magnitude of vector
magnitude_w = np.linalg.norm(w) #it will give the magnitude of orignal vector with length 1
magnitude1_w = np.linalg.norm(w, 1)
print(magnitude_w)
print(magnitude1_w)


# pytorch(torch) : torch implementation
v_tensor = torch.tensor([5.0, -3.1])
v_magnitude = torch.norm(v_tensor)
v_norm = v_tensor / v_magnitude
print(v_tensor)
print(f"normalized vector: {v_norm}")
print(f"magnitude of vector: {torch.norm(v_norm)}")
