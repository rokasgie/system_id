import csv
import argparse
from lmfit import minimize, fit_report
import numpy as np
from lmfit_model import model, Timestep, State, Controls, get_parameters
import pandas as pd
from pandas import DataFrame

parser = argparse.ArgumentParser(description="Simulation vehicle model verification")
parser.add_argument("--file-path", type=str, required=True, help="Path to data file")
parser.add_argument("--skip-first", type=int, default=1000, help="Skip first n lines")
parser.add_argument("--no-samples", type=int, default=50000, help="Number of samples to use")
parser.add_argument("--use-every", type=int, default=4, help="Use every nth sample")
# parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
# parser.add_argument("--cuda", action='store_true', help="Use cuda")
args = parser.parse_args()


def getFloatArray(x):
    return np.array([float(i) for i in x])


def filter_data(df):
    df_filtered = df
    # df_filtered = df[df.current_v_x > 1]
    df_filtered = df_filtered[df_filtered.current_v_y / df_filtered.current_v_x < 0.75]
    df_filtered = df_filtered[df_filtered.current_v_x < df_filtered.current_v_x.quantile(0.98)]
    df_filtered = df_filtered[df_filtered.current_v_y < df_filtered.current_v_y.quantile(0.98)]
    df_filtered = df_filtered[df_filtered.current_r < df_filtered.current_r.quantile(0.98)]

    df_filtered = df_filtered[::3]
    return df_filtered


def get_timesteps1(csv_file):
    df = pd.read_csv(csv_file)

    df_filtered = filter_data(df)

    curr_state = State(v_x=getFloatArray(df_filtered['current_v_x']), v_y=getFloatArray(df_filtered['current_v_y']),
                       r=getFloatArray(df_filtered['current_r']), yaw=getFloatArray(df_filtered['current_yaw']))

    next_state = State(v_x=getFloatArray(df_filtered['next_v_x']), v_y=getFloatArray(df_filtered['next_v_y']),
                       r=getFloatArray(df_filtered['next_r']), yaw=getFloatArray(df_filtered['next_yaw']))

    controls = Controls(delta=getFloatArray(df_filtered['delta']), dc=getFloatArray(df_filtered['acc']))

    timestep = Timestep(curr_state, next_state, controls, getFloatArray(df_filtered['dt']))

    return timestep


def get_timesteps(csv_file, skip, no_samples, take_every):
    reader = csv.DictReader(open(csv_file))
    data = {}
    for name in reader.fieldnames:
        data[name] = []

    i, j = 0, 0
    for row in reader:
        if i < skip:
            i += 1
            continue
        elif len(data[reader.fieldnames[0]]) < no_samples:
            if j == take_every:
                j = 1
                for key, value in row.items():
                    data[key].append(value)
            else:
                j += 1
        else:
            break

    curr_state = State(v_x=getFloatArray(data['current_v_x']), v_y=getFloatArray(data['current_v_y']),
                       r=getFloatArray(data['current_r']))

    next_state = State(v_x=getFloatArray(data['next_v_x']), v_y=getFloatArray(data['next_v_y']),
                       r=getFloatArray(data['next_r']))

    controls = Controls(delta=getFloatArray(data['delta']), dc=getFloatArray(data['acc']))

    timestep = Timestep(curr_state, next_state, controls, getFloatArray(data['dt']))

    return timestep


if __name__ == "__main__":
    parameters = get_parameters()
    data = get_timesteps1(args.file_path)
    print("Running with {} samples".format(len(data.curr_state.v_x)))
    result = minimize(model, parameters, args=(data.curr_state, data.next_state, data.controls, data.dt))
    print(fit_report(result))
