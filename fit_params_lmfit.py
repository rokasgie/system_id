import csv
import argparse
from lmfit import minimize, fit_report, Parameter
import numpy as np
from lmfit_model import model
import pandas as pd

parser = argparse.ArgumentParser(description="Simulation vehicle model verification")
parser.add_argument("--file-path", type=str, required=True, help="Path to csv data file")
parser.add_argument("--n-data-to-use", type=int, required=True, help="Number of data points to use")
args = parser.parse_args()


def getFloatArray(x):
    return np.array([float(i) for i in x])


def filter_data(df):
    df_filtered = df
    df_filtered = df[df.current_v_x > 1]
    df_filtered = df_filtered[np.abs(df_filtered.slip) < 0.75]
    df_filtered = df_filtered[df_filtered.current_v_x < df_filtered.current_v_x.quantile(0.98)]
    df_filtered = df_filtered[df_filtered.current_v_y < df_filtered.current_v_y.quantile(0.98)]
    df_filtered = df_filtered[df_filtered.current_r < df_filtered.current_r.quantile(0.98)]

    # df_filtered = df_filtered[::3]
    return df_filtered


def get_timesteps1(csv_file):
    df = pd.read_csv(csv_file)
    df["slip"] = -np.arctan2(df.current_v_y, np.abs(df.current_v_x))

    df_filtered = filter_data(df)

    curr_state = State(v_x=getFloatArray(df_filtered['current_v_x']), v_y=getFloatArray(df_filtered['current_v_y']),
                       r=getFloatArray(df_filtered['current_r']), yaw=getFloatArray(df_filtered['current_yaw']))

    next_state = State(v_x=getFloatArray(df_filtered['next_v_x']), v_y=getFloatArray(df_filtered['next_v_y']),
                       r=getFloatArray(df_filtered['next_r']), yaw=getFloatArray(df_filtered['next_yaw']))

    controls = Controls(delta=getFloatArray(df_filtered['delta']), dc=getFloatArray(df_filtered['acc']))

    timestep = Timestep(curr_state, next_state, controls, getFloatArray(df_filtered['dt']))

    return timestep


if __name__ == "__main__":
    # Initilaise paramters
    parameters = [
        # Gravity is constant, keep this fixed
        Parameter('INERTIA_G_', value=9.81, vary=False),
        Parameter('INERTIA_M_', value=190, vary=True, min=100, max=300),
        Parameter('INERTIA_I_Z_', value=110, min=0, vary=True),

        Parameter('AERO_C_DOWN_', value=1.9032, min=0, vary=True),
        Parameter('AERO_C_DRAG_', value=0.7, min=0, vary=True),

        Parameter('KINEMATIC_L_', value=1.53, vary=False),
        Parameter('KINEMATIC_L_F_', value=1.22, vary=False),
        Parameter('KINEMATIC_L_R_', value=1.22, vary=False),
        Parameter('KINEMATIC_W_FRONT_', value=0.5, vary=False),

        Parameter('FRONT_AXLE_WIDTH_', value=1.2, vary=False),
        Parameter('REAR_AXLE_WIDTH_', value=1.2, vary=False),

        Parameter('TIRE_B_', value=12.56, vary=True),
        Parameter('TIRE_C_', value=-1.38, vary=True),
        Parameter('TIRE_D_', value=1.6, vary=True),
        Parameter('TIRE_E_', value=-0.58, vary=True),

        # These parameter control the acceleration of the car so we leave them fixed
        Parameter('DRIVETRAIN_CM1_', value=5000, min=0, vary=False),
        Parameter('DRIVETRAIN_CR0_', value=180, min=0, vary=False),
        Parameter('DRIVETRAIN_M_LON_', value=30, vary=False)
        # cType.m_lon_add = cType.nm_wheels * cType.inertia / (cType.r_dyn * cType.r_dyn);
    ]

    # Read data
    data = pd.read_csv(args.file_path)

    # Remove empty columns
    data = data.dropna(axis=1)

    # Make the index to be the timestamp
    data.index = pd.to_datetime(data.rosbagTimestamp.values)
    dt = data.dt.mean()

    # Remove not interesting/redundant columns
    data = data.drop(["dt", "seq", "secs", "nsecs", "frame_id", "rosbagTimestamp", "front_slip_angle", "rear_slip_angle", "next_v_x", "next_v_y", "next_r", "next_yaw"], axis=1)

    # Add slip angle
    data["slip"] = -np.arctan2(data.current_v_y, np.abs(data.current_v_x))

    # Calculate linear derivatives
    data['v_x_dot'] = np.hstack(((data.current_v_x.values[1:] - data.current_v_x.values[:-1])*(1/dt), np.nan))
    data['v_y_dot'] = np.hstack(((data.current_v_y.values[1:] - data.current_v_y.values[:-1])*(1/dt), np.nan))
    data['r_dot'] = np.hstack(((data.current_r.values[1:] - data.current_r.values[:-1])*(1/dt), np.nan))

    # Remove last row as it's a nan now
    data = data.dropna()

    # Filter bad data
    print("Data entires before filtering", len(data))
    data = data[np.abs(data.slip) < 0.75]
    data = data[data.current_v_x > 1.0]

    # Filter outlier data from v_y_dot and r_dot axis
    q0 = data.v_y_dot.quantile(0.025)
    q1 = data.v_y_dot.quantile(0.975)
    data = data[q0 < data.v_y_dot]
    data = data[data.v_y_dot < q1]
    print("Data entires after filtering", len(data))

    # Save derivatives in a different data frame
    data_dot = data[['v_x_dot', 'v_y_dot', 'r_dot']].copy()

    # Rename columns for convinience
    renames = {"current_v_x": "v_x",
            "current_v_y": "v_y",
            "current_r": "r",
            "current_yaw": "yaw",
            "delta": "delta_cmd",
            "acc": "acc_cmd"}
    data = data.rename(renames, axis=1)

    # Use only a subset of all data to make this finish in a sensible time
    data = data[:args.n_data_to_use]
    data_dot = data_dot[:args.n_data_to_use]
    print("Using only first {} data entries".format(args.n_data_to_use))

    # Fit model
    print("Starting fitting...")
    result = minimize(model, parameters, args=(data, data_dot))
    print(fit_report(result))

    print("\n------ RESIDUALS -------")
    print(result.residual.mean())
