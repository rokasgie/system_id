import torch
from torch.nn import Module
from torch.nn import Linear


class Model(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = Linear(input_size, output_size)
        self.linear2 = Linear(input_size, output_size)

    def forward(self, x1, x2):
        # inp = torch.cat((x1, x2))
        # return self.linear1(inp)
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        return out1, out2
