# Broadcasting means pytorch automatically expands a smaller tensors so an operation can be performed with a larger tensor without you manually copying the values.
#  think of it like an scalar operation we did 


#----------------------- BROADCASTING A 1D TENSOR ------------------------------------
import torch 
x = torch.tensor([1,2,3])
print(x+10)

# You get:
# tensor([11, 12, 13])

# But 10 is only one number.

# Conceptually, PyTorch treats it like:
# [1, 2, 3]
# +
# [10, 10, 10]
# ↓
# [11, 12, 13]

# You didn't actually create [10, 10, 10]; PyTorch handles the broadcasting internally.

# -----------------------BROADCASTING A 2D TENSOR--------------------------------------
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

y = torch.tensor([10, 20, 30])
print(x+y)


# mathi x ko shape -> 2 x 3
# mathi y ko shape -> 1 x 3 

# [1  2  3]     [10 20 30]
# [4  5  6]  +  [10 20 30]
# ↓
# [11 22 33]
# [14 25 36]

# The broadcasting rule 
# When PyTorch compares dimensions, starting from the rightmost dimension, two dimensions are compatible if:

# They are equal, or
# One of them is 1.

# For example:

# (3, 1)
# (3, 3)

# works because:

# 1 can expand to 3

# And:

# (2, 3)
# (3)

# works because PyTorch treats the second as:

# (1, 3)

# and then:

# (2, 3)
# (1, 3)

# works.

# ----------------------------------------------------------------------------------
# 6. When broadcasting fails

# Consider:

# (2, 3)
# (2, 4)

# Compare from the right:

# 3
# 4

# They aren't equal, and neither is 1.

# Therefore:

# Cannot broadcast.

# This will produce an error.