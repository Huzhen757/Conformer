import torch

a = [256, 256]
for i, j in zip([256] + a, a + [256]):
    print([i, j])
    
b = [256, 256]
for i, j in zip([256] + b, b + [4]):
    print([i, j])

# h = 4
# w = 3
# grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
# print("grid_y: ")
# print(grid_y)
# print("grid_x: ")
# print(grid_x)
# grid = torch.stack((grid_x, grid_y), 2).float()
# print("after stack: ")
# print(grid)

point = torch.rand((100, 8, 2))
grid = torch.rand((2142, 8, 2))
point.unsqueeze(1)
grid.unsqueeze(0)
distance = (point - grid).pow(2)

