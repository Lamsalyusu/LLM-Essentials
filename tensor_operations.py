import torch 
x = torch.tensor([1,2,3])
y = torch.tensor([10,20,30])
# it operates element by element 
# print(x+y)
# result --> tensor([11,22,33])

# -----------------------------------------------------------------------------------

# print(x-y)
# result --> [1-10,2-20,3-30]

# -----------------------------------------------------------------------------------


# when we do x * you are doing element-wise multiplication not matrix multiplication 
# print(x * y)
# [1,2,3] * [10,20,30] ---> tensor([10,40,90])

# -----------------------------------------------------------------------------------

# print(x / y) 
# tensor([0.1000,0.1000,0.1000])

# ----------------------------------------------------------------------------------

# + → element-wise addition
# - → element-wise subtraction
# * → element-wise multiplication
# / → element-wise division

# ----------------------------SCALAR OPERATION--------------------------------------

#  x yahan mathi bata referencing bhairako chha
# x bhaneko --> 1 ,2,3 ho yesma ani scalar operation ma harek element ma +10 add hunxa
#  so 1 + 10, 2+10 ,3+10
# print(x+10)
#  same in case of mulitply also x * 2 --> 1/2,2/2,3/2 


# ------------------Aggregation Operations------------------------------------------

x = torch.tensor([1,2,3,4])
# print(x.sum())
# result --> tensor(10)

# -----------------------------------------------------------------------------------

# yo jun maile .float().mean() gareko chhu no tyo bhaneko mean in  float ma nikalne bhaneko ho 
#  if maile directly .mean() use gare bhane runtime error dekhauchha so dtype define garerai mean nikalnu parchha
# print(x.float().mean())
# result --> tensor(2.5000) aauxa yesko answer 

# -----------------------------------------------------------------------------------

# print(x.max())  
# result --> tensor(4)

# -----------------2D Tensor Operations----------------------------------------------

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# shape : (2,3)
# x.sum() --> yesko kaam chai sabai elements lai add up gardine ho
# print(x.sum())

# ------------------------------------------------------------------------------------

# print(x.sum(dim=0))
# result --> tensor([5,7,9])
# dim = 0 bhaneko tala dekhako jastai 
# 1  2  3
# 4  5  6
# ↓  ↓  ↓
# 5  7  9

# ------------------------------------------------------------------------------------

# print(x.sum(dim = 1))
# result --> tensor([6, 15])
# 1 + 2 + 3 = 6
# 4 + 5 + 6 = 15

# ------------------------------------------------------------------------------------

# dim=0 → operate vertically/down rows
# dim=1 → operate horizontally/across columns

# ----------------MATRIX MULTIPLICATION------------------------------------------------
x = torch.tensor([
    [1, 2],
    [3, 4]
])

y = torch.tensor([
    [5, 6],
    [7, 8]
])
# yo bhayo element wise multiplication
# print(x * y)

#  yo bhayo matrix multiplication 
# print(x @ y)

#  yo bhayo transpose (.T)
print(x.T)
