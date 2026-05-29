import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


model = TinyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter("runs/phase0_lesson12")

x_train = torch.randn(256, 10)
y_train = torch.randint(0, 2, (256,))

x_val = torch.randn(256, 10)
y_val = torch.randint(0, 2, (256,))

for step in range(100):
    optimizer.zero_grad()

    outputs = model(x_train)
    train_loss = criterion(outputs, y_train)

    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val)

    writer.add_scalar("loss/train", train_loss.item(), step)
    writer.add_scalar("loss/val", val_loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()

print("Wrote TensorBoard logs to runs/phase0_lesson12")
