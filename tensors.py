# Tensors are specialized data structures that are very similar to arrays and matrices.
# In pytorch we use tensors to encode the inputs and outputs of the model as welll as the models parameter 
# Tensors are similar to NumPy's ndarraays except that tensors can run on GPU  and other hardware accelerators 
# A tensors is simply a container for numbers arranged in different dimensions.

# NUmber --> 0D tensors
# [1,2,3] --> 1D tensors
# [1,2][2,4] ---> 2D tensors
# multiple 2D table --> 3D tensors

import torch
import numpy as np
import array

# 0D tensors
data0 = 1
x_data0 = torch.tensor(data0)
# print("0D Tensor ", x_data0)
# print("0D Tensor ", x_data0.shape)

# 1D Tensors
data1 = [1,2,3,4]
x_data1 = torch.tensor(data1)
# print("1D Tensor ", x_data1)
# print("1D Tensor ", x_data1.shape)

# 2D Tensors
data2 = [[1,2],[3,4]]
x_data2 = torch.tensor(data2)
# print("2D Tensor ", x_data2)
# print("2D Tensor ", x_data2.shape)

# 3D tensors 
data3 = [[[1,2,3],[3,4,5]],[[1,2,3],[4,5,6]]]
x_data3 = torch.tensor(data3)
# print("3d Tensor ", x_data3)
# print("3D Tensor ",x_data3.shape)

# (2,2,3) --> means 2 tables 2 rows and 3 columns 
# for a grayscale image --> 28 x 28 grayscale image --> 28 rows x 28 columns  --> tensor shape --> (28,28)
# for a color RGB image ---> 28 x 28 x 3 --> 28 pixels high x 28 pixels wide x 3 color channels --> tesnor shape = (28,28,3)


# From a NumPy array 
data = [[1,2],[3,4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print("from numpy array: ", x_np)


x_data = torch.tensor(data)
x_ones = torch.ones_like(x_data) 
# shape and datatypes retains doesnot changes unless explicitly overridden like below
# jastai aba x_data ko shape 2x2 chha ra data type integer chha bhane .ones_like function le chai teslai tei shape ra data type ko 1s ma convert gardinchha
# print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) 
# overrides the datatype of x_data
# yesma chai shape tei chha but the datatype has been changed 
# print(f"Random Tensor: \n {x_rand} \n")


# With random or constant values:
# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

shape = (2,3)
# print(f"Random Tensor: \n {torch.rand(shape)} \n")
# print(f"Ones Tensor: \n {torch.ones(shape)} \n")
# print(f"Zeros Tensor: \n {torch.zeros(shape)}")

# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")


x= torch.tensor([
    [10,20,30],
    [40,50,60]
])

#        column
#        0   1   2
#      ┌───────────
# row 0│ 10  20  30
# row 1│ 40  50  60

# x[0] ---> give me row 0 --> tensor ([10,20,30])
# x[1] ---> give me row 1 --> tesnor*([40,50,60])

# specifying both row and columns:
# x[0,1] ---> meaning row 0 and column 1 --> result 20

# ----------------------------------------------------------------------------------------

# Slicing gets a portion/range 
# the basic syntax is [start:stop]
# important: stop is not included 


x = torch.tensor([10, 20, 30, 40, 50])
# print(x[1:4])
#  gives --> tensor([20, 30, 40])

# index:  0   1   2   3   4
# value: 10  20  30  40  50
#             ↑       ↑
#           start    stop

# omitting start or stop
#  start from the beginning and go until index 3 
print(x[:3])
# result --> 10 20 30 

# start at index 2 and go to the end
print(x[2:])
#  result 30, 40 ,50


# give me everything 
print(x[:])
# result 10, 20, 30, 40, 50

# ------------------------------------------------------------------------------------

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# shape : 3 x 3 

# get first two rows 
print(x[:2])
# result 
# 10 20 30
# 40 50 60

# get row 1 and row 2  (since it starts with 0 so )
print(x[1:])
# result 
# 40 50 60
# 70 80 90

# ----------------------------------------------------------------------------------
# getting specific columns 
# x [row ,columns]
# : --> means everything in that dimension
#  so x[:,0] --> give me all rows but only column 0
print(x[:,0])
# result --> tensor ([10,40 ,70])
# 10  20  30
# ↑
# 40  50  60
# ↑
# 70  80  90
# -----------------------------------------------------------------------------------

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

#  shape : 2,3
# 2 rows  3 columns
#  contains 6 numbers 
# we can reshape it : 
x.reshape(3,2)

# for reshaping follow these rules while reshaing the total numbers of items must not changes
# (1, 6) → 6
# (2, 3) → 6
# (3, 2) → 6
# (6, 1) → 6

# -----------------------------------------------------------------------------------

x = torch.tensor([1, 2, 3, 4, 5, 6])
y =x.reshape(2,-1)
# This doesn't modify x in place. It returns a new reshaped tensor.
print(y)
#  yo -1 ko kura k ho bhanda 
# PyTorch figures out the missing dimension.

# It knows:
# 6 total elements
# 2 rows

# Therefore:
# 6 / 2 = 3

# So:
# (2, -1)
# becomes:
# (2, 3)

# The -1 means:
# "PyTorch, you calculate this dimension for me."