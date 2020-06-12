import csv
import argparse
from lmfit import minimize, fit_report
import numpy as np
from lmfit_model import model, Timestep, State, Controls, get_parameters
import pandas as pd

parser = argparse.ArgumentParser(description="Simulation vehicle model verification")
parser.add_argument("--file-path", type=str, required=True, help="Path to csv data file")
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


def get_timesteps(csv_file):
    df = pd.read_csv(csv_file)

    df_filtered = filter_data(df)

    curr_state = State(v_x=getFloatArray(df_filtered['current_v_x']), v_y=getFloatArray(df_filtered['current_v_y']),
                       r=getFloatArray(df_filtered['current_r']), yaw=getFloatArray(df_filtered['current_yaw']))

    next_state = State(v_x=getFloatArray(df_filtered['next_v_x']), v_y=getFloatArray(df_filtered['next_v_y']),
                       r=getFloatArray(df_filtered['next_r']), yaw=getFloatArray(df_filtered['next_yaw']))

    controls = Controls(delta=getFloatArray(df_filtered['delta']), dc=getFloatArray(df_filtered['acc']))

    timestep = Timestep(curr_state, next_state, controls, getFloatArray(df_filtered['dt']))

    return timestep


if __name__ == "__main__":
    parameters = get_parameters()
    data = get_timesteps(args.file_path)
    print("Running with {} samples".format(len(data.curr_state.v_x)))
    result = minimize(model, parameters, args=(data.curr_state, data.next_state, data.controls, data.dt))
    print(fit_report(result))
