# Vector Dot Product
# The dot product is one of the MOST important operations in Machine Learning.
# It multiplies corresponding elements of two vectors and then adds them.

# formula => [a,b]⋅[x,y]
# This is the holy grail of ML operations. It multiplies matching elements and adds the results together.
# It’s used to calculate the weighted sum in neural network layers ($z = \vec{w} \cdot \vec{x}$).

import numpy as np
v1 = np.array([1, 2])
v2 = np.array([4, 5])
# pure pythonic way

v_dp = [sum(a * b for a, b in zip(v1, v2))]
print(f"pythonic way of dot product is: {v_dp}")

# numpy way
v_dot = np.dot(v1, v2) #numpy method 1
v_dot2 = v1.dot(v2) #numpy method 2
v_dot3 = v1 @ v2 # numpy method 3 /# The '@' operator is the modern standard for matrix/vector multiplication
print(f"the dot product of two vector is: {v_dot}")
print(f"the dot product of two vector is: {v_dot2}")
print(f"the dot product of two vector is: {v_dot3}") ## (1*4) + (2*5) = 3 + 8 = 11

# The dot product is everywhere in Machine Learning.
# 1. Predictions in ML Models

x = np.array([1, 2, 3])
w = np.array([0.5, 0.3, 0.1])

prediction = np.dot(x, w)
print(f"prediction: {prediction}") 