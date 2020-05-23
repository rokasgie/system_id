import torch

DEVICE = torch.device('cuda')


def tensor(val):
    return torch.tensor(float(val), requires_grad=True, device=DEVICE)