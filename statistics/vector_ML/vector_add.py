# . Addition (Combine Information)
# In ML, you use this when adding a bias term to your inputs, or when combining different feature vectors.
# Add two feature vectors element-wise.
#When we add vectors, we are usually:

# Combining information
# Updating knowledge
# Moving toward a better prediction
# Adjusting parameters 


import numpy as np
# Data poin (feature 1  and feature 2)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# pure pythonic way using loop
# added_zip = [a + b for a, b in zip(v1, v2)]
# vector_added = [v1[i] + v2[i] for i in range(len(v1))]

# numpy way
v_addition = v1 + v2
# print(f"pythonic way of v addition is: {vector_added}") output =[np.int64(5), np.int64(7), np.int64(9)]
print(f"numpy way of v addition is: {v_addition}")
# print(added_zip) output =  [np.int64(5), np.int64(7), np.int64(9)]