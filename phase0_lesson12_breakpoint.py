import torch

x = torch.randn(4, 3)
y = torch.randn(4, 3)

loss = (x / 0).mean()

if torch.isnan(loss) or torch.isinf(loss):
    breakpoint()

print("loss:", loss)
