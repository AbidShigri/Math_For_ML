# import tensorflow as tf

# # 1. Creating two matrices (Tensors)
# # Matrix A: Shape (2, 3)
# matrix_a = tf.constant([[1, 2, 3], 
#                         [4, 5, 6]], dtype=tf.float32)

# # Matrix B: Shape (3, 2)
# matrix_b = tf.constant([[7, 8], 
#                         [9, 1], 
#                         [2, 3]], dtype=tf.float32)

# # 2. Performing Matrix Multiplication
# # In ML, this operation is often called a "dot product" or "matmul"
# result = tf.matmul(matrix_a, matrix_b)

# print(result)