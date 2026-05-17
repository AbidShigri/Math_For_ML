# 2. Subtraction (Difference / Error)In ML, subtraction is heavily used to calculate error or loss
#  ($\text{Actual} - \text{Predicted}$) and to update model weights in Gradient Descent.

# ML Relevance
# 1. Error Calculation
# A model predicts values.
# by subtactiing predicted from actual

# 2. Gradient Descent (Learning)

# This is one of the MOST important uses.
# Models learn by subtracting gradients from weights.
# Basic formula:w new = w old - −η∇L

# Distance Between Data Points
# In ML, we often measure similarity.


import numpy as np

# data Point (vector 1 and vector 2) feautures

v1 = np.array([4, 3, 5])
v2 = np.array([2, 6, 3])

# pure pythonic way
vec_sub = [v1[i] - v2[i] for i in range(len(v1))]
vec_sub_z = [i - j for i, j in zip(v1, v2)]

# numpy way
v_subtraction = v1 - v2

print(f"subtraction of vector is: {vec_sub}")
print(f"subtraction of vector is: {vec_sub_z}")
print(f"subtraction of vector is: {v_subtraction}")
print(f"no of element in each dimension: {v_subtraction.shape}")
print(f"this vector dimension is: {v_subtraction.ndim}")

