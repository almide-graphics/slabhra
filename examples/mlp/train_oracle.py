"""Oracle for the slabhra training example: train the tiny MLP with Adam to
fit a fixed (x, target) and report the loss curve. slabhra must track it.

    python train_oracle.py
"""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# fixed deterministic init (same as examples/mlp/train.almd)
W1 = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], requires_grad=True)
b1 = torch.tensor([[0.01, -0.02, 0.03]], requires_grad=True)
W2 = torch.tensor([[0.7, -0.8], [0.9, 0.1], [-0.2, 0.5]], requires_grad=True)
b2 = torch.tensor([[0.05, -0.06]], requires_grad=True)
x = torch.tensor([[0.5, -1.0]])
target = 0  # train to predict class 0

opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.1)
for step in range(101):
    opt.zero_grad()
    z = torch.relu(x @ W1 + b1) @ W2 + b2
    loss = torch.nn.functional.cross_entropy(z, torch.tensor([target]))
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.6f}")
    loss.backward()
    opt.step()
