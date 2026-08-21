# import torch 
# x= torch.tensor(2.0,requires_grad=True)
# y = x ** 2
# y.backward()
# print(x.grad)


# A gradient tells us the direction and amount a value needs to change to affect the output.

# require_grad = True --> this means "Keep track of operations involving x because later I want the gradient."

# y.backward() --> "Calculate the gradient of y with respect to the variables that require gradients."

# import torch

# x = torch.tensor(2.0, requires_grad=True)

# y = x ** 2

# print("x:", x)
# print("y:", y)

# y.backward()

# print("Gradient:", x.grad)


# import torch

# x = torch.tensor(2.0)
# target = torch.tensor(10.0)

# w = torch.tensor(3.0, requires_grad=True)
# b = torch.tensor(1.0, requires_grad=True)

# prediction = w * x + b

# loss = (prediction - target) ** 2

# print("Prediction:", prediction)
# print("Loss:", loss)

# loss.backward()

# print("Weight gradient:", w.grad)
# print("Bias gradient:", b.grad)