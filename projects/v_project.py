# This project is a small Machine Learning-inspired system developed using Python and NumPy. The main purpose of the project is to understand how vectors and scalar operations are used in real-world AI systems.

# In this project, each user is represented as a vector. Every value inside the vector represents a feature, skill, or interest of that user. For example:

# [9,8,7]

# may represent:

# [Python Skill, Math Skill, AI Skill]

# The system compares these vectors using cosine similarity to determine which users are most similar to each other.

# Cosine similarity is calculated using:

# dot product
# vector magnitude
# normalization concepts

import numpy as np
# here this is a data set 
# each user represent as a vector
# and each value reprent as a feature,or vector feature
user = {
    "Ali": np.array([10, 10, 9]),
    "Sara": np.array([5, 9, 7]),
    "Abid": np.array([9, 9, 10]),
    "Rehana": np.array([8, 9, 9]),
    "Fatima": np.array([9, 8, 8]),
    "Muslim": np.array([9, 8, 8])
}

# target user
target_user = "Ali"
# get target vectro
target_vector = user[target_user]
# Function of cosine similarity
# Cosine similarity is a method used to measure how similar two vectors are.

# Instead of comparing only the values, it compares the direction of vectors.

# If two vectors point in almost the same direction, their cosine similarity is close to:

# 1

# If they are very different, the value becomes closer to:

# 0

def cosine_similarity(v1, v2):
    # dot_product use here to direction
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

# normalization help make data comparable or stable

    similarity = dot_product / (magnitude_v1 * magnitude_v2)
    return similarity

# variable to store best match
best_match = None
highest_score = -1

# compare with all user

for name , vector in user.items():
    if name != target_user:
        score = cosine_similarity(target_vector, vector)
        print(name, "->" ,score)
        if score > highest_score:
            highest_score = score
            best_match = name

# final ouput
print(f"best match is: {best_match}")
print(f"Similarity score is: {highest_score}")