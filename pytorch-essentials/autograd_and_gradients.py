import torch 
x= torch.tensor(2.0,requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)


# A gradient tells us the direction and amount a value needs to change to affect the output.

# require_grad = True --> this means "Keep track of operations involving x because later I want the gradient."

# y.backward() --> "Calculate the gradient of y with respect to the variables that require gradients."