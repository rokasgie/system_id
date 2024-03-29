import sys
import argparse
import csv
import torch
from torch.optim import Adam, SGD, LBFGS, RMSprop
from torch.utils.data import DataLoader

from fssim_model import Constants, State, Controls
from dataset import TimestepDataset

parser = argparse.ArgumentParser(description="Simulation vehicle model verification")
parser.add_argument("--file-path", type=str, required=True, help="Path to data file")
parser.add_argument("--skip-first", type=int, default=1000, help="Skip first n lines")
parser.add_argument("--no-samples", type=int, default=50000, help="Number of samples to use")
parser.add_argument("--use-every", type=int, default=4, help="Use every nth sample")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--cuda", action='store_true', help="Use cuda")
args = parser.parse_args()


if __name__ == "__main__":
    constants = Constants()
    dataset = TimestepDataset(args, constants)
    # dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    # model = TimestepModel(constants)

    optimizer = Adam(constants.as_list(), lr=0.1)
    criterion = torch.nn.MSELoss()

    loss = float('inf')
    epochs = 100
    epoch = 0
    print("Training started with {} samples".format(len(dataset)))
    while loss != 0 and epoch < epochs:
        for timestep in dataset:
            # def closure():
                # optimizer.zero_grad()
                # loss = model(timesteps, criterion)
                # print("Constants: {}".format(constants))
                # print("Loss: {}".format(loss))
                # loss.backward()
                # return loss
            # optimizer.step(closure)

            # print(timestep)
            loss = timestep.forward(criterion)
            # print("Constants: {}".format(constants))
            # print("Loss: {}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {}, loss: {}, constants: {}".format(epoch, loss, constants))
        epoch += 1
