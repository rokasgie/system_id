import torch

DEVICE = torch.device('cuda')


def tensor(val):
    if type(val) == int or type(val) == str:
        val = float(val)
    return torch.tensor(val, requires_grad=True, device=DEVICE)