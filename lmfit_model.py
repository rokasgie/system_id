import numpy as np
from lmfit import Parameters, Parameter
from dataclasses import dataclass
from typing import List

@dataclass()
class State:
    #x: np.array # float
    #y: np.array # float
    r: np.array # float
    yaw: np.array # float

    v_x: np.array # float
    v_y: np.array # float

    def __add__(self, other):
        return State(v_x=self.v_x + other.v_x, v_y=self.v_y + other.v_y,
                     r=self.r + other.r, yaw=self.yaw + other.yaw)

    def __mul__(self, val):
        return State(v_x=self.v_x * val, v_y=self.v_y * val,
                     r=self.r * val, yaw=self.yaw * val)


@dataclass()
class Controls:
    dc: np.array # float
    delta: np.array # float


@dataclass()
class Timestep:
    curr_state: State
    next_state: State
    controls: Controls
    dt: np.array


parameters = Parameters()
params = \
[
    Parameter('INERTIA_G_', value=9.81, vary=False),
    Parameter('INERTIA_M_', value=190, vary=False),
    Parameter('INERTIA_I_Z_', value=110, min=0),
    Parameter('AERO_C_DOWN_', value=1.9032, min=0),
    Parameter('AERO_C_DRAG_', value=0.7, min=0),
    Parameter('KINEMATIC_L_', value=1.53, vary=False),
    Parameter('KINEMATIC_L_F_', value=1.22, vary=False),
    Parameter('KINEMATIC_L_R_', value=1.22, vary=False),
    Parameter('KINEMATIC_W_FRONT_', value=0.5, vary=False),
    Parameter('FRONT_AXLE_WIDTH_', value=1.2, vary=False),
    Parameter('REAR_AXLE_WIDTH_', value=1.2, vary=False),
    Parameter('TIRE_B_', value=12.56),
    Parameter('TIRE_C_', value=-1.38),
    Parameter('TIRE_D_', value=1.6),
    Parameter('TIRE_E_', value=-0.58),
    Parameter('DRIVETRAIN_CM1_', value=5000, min=0),
    Parameter('DRIVETRAIN_CR0_', value=180, min=0),
    Parameter('DRIVETRAIN_M_LON_', value=29.98, min=0)
]


def get_parameters():
    for p in params:
        parameters.add(p)
    return parameters


def getFDown(params: Parameters, curr_state: State):
    return params['AERO_C_DOWN_'] * curr_state.v_x ** 2


def getFz(params: Parameters, curr_state: State):
    return params['INERTIA_G_'] * params['INERTIA_M_'] + getFDown(params, curr_state)


def getSlipAngle(params: Parameters, curr_state: State, controls: Controls, isFront=True):
    lever_arm_len = params['KINEMATIC_L_'] * params['KINEMATIC_W_FRONT_']
    v_x = np.maximum(1, curr_state.v_x)
    if isFront:
        return np.arctan((curr_state.v_y + lever_arm_len * curr_state.r) /
                       (v_x - 0.5 + params['FRONT_AXLE_WIDTH_'] * curr_state.r)) \
               - controls.delta
    else:
        return np.arctan((curr_state.v_y - lever_arm_len * curr_state.r) /
                       (v_x - 0.5 + params['REAR_AXLE_WIDTH_'] * curr_state.r))


def getDownForceFront(params: Parameters, Fz):
    return 0.5 * params['KINEMATIC_W_FRONT_'] * Fz


def getFy(params: Parameters, curr_state: State, controls: Controls, isFront=True):
    Fz = getFz(params, curr_state)
    slipAngle = getSlipAngle(params, curr_state, controls, isFront)
    Fz_axle = getDownForceFront(params, Fz)
    mu = params['TIRE_D_'] * np.sin(params['TIRE_C_'] * np.arctan(params['TIRE_B_'] *
                                            (1 - params['TIRE_E_']) * slipAngle + params['TIRE_E_']
                                                                * np.arctan(params['TIRE_B_'] * slipAngle)))
    return Fz_axle * mu


def getFyF(params: Parameters, curr_state: State, controls: Controls):
    return getFy(params, curr_state, controls, True)


def getFyR(params: Parameters, curr_state: State, controls: Controls):
    return getFy(params, curr_state, controls, False)


def getFdrag(params: Parameters, curr_state: State):
    return params['AERO_C_DRAG_'] * curr_state.v_x ** 2


def getFx(params: Parameters, curr_state: State, controls: Controls):
    return controls.dc * params['DRIVETRAIN_CM1_'] - getFdrag(params, curr_state) - params['DRIVETRAIN_CR0_']


def kinCorrection(params: Parameters, curr_state: State, controls: Controls, dt, next_dyn_state: State):
    v_x_dot = getFx(params, curr_state, controls) / (params['INERTIA_M_'] + params['DRIVETRAIN_M_LON_'])
    v = np.sqrt(curr_state.v_x ** 2 + curr_state.v_y ** 2)
    v_blend = 0.5 * (v - 1.5)
    blend = np.max(np.minimum(1, v_blend), 0)

    next_dyn_state.v_x = blend * next_dyn_state.v_x + (1 - blend) * (curr_state.v_x + dt * v_x_dot)

    v_y = np.tan(controls.delta) * next_dyn_state.v_x * params['KINEMATIC_L_R_'] / params['KINEMATIC_L_']
    r = np.tan(controls.delta) * next_dyn_state.v_x / params['KINEMATIC_L_']

    next_dyn_state.v_y = blend * next_dyn_state.v_y + (1 - blend) * v_y
    next_dyn_state.r = blend * next_dyn_state.r + (1 - blend) * r

    return next_dyn_state


def model(params, curr_state: State, next_state: State, controls: Controls, dt):
    # x_dot = np.cos(curr_state.yaw) * curr_state.v_x - np.sin(
    #     curr_state.yaw) * curr_state.v_y
    # y_dot = np.sin(curr_state.yaw) * curr_state.v_x + np.cos(
    #     curr_state.yaw) * curr_state.v_y

    yaw_dot = curr_state.r

    # Changed yaw to r in this equation
    v_x_dot = (curr_state.r - curr_state.v_y) - \
              (getFx(params, curr_state, controls) - np.sin(curr_state.yaw) * 2 * getFyF(params, curr_state, controls)) / \
              (params['INERTIA_M_'] + params['DRIVETRAIN_M_LON_'])

    v_y_dot = (np.cos(controls.delta) * 2 * getFyF(params, curr_state, controls) + 2 * getFyR(params, curr_state, controls)) / \
              params['INERTIA_M_'] - curr_state.r * curr_state.v_x
    r_dot = ((np.cos(controls.delta) * 2 * getFyF(params, curr_state, controls) * params['KINEMATIC_L_F_']) -
             (2 * getFyR(params, curr_state, controls) * params['KINEMATIC_L_R_'])) / params['INERTIA_I_Z_']

    dyn_state = State(v_x_dot, v_y_dot, r_dot, yaw_dot)
    next_dyn_state = curr_state + dyn_state * dt
    final_state = kinCorrection(params, curr_state, controls, dt, next_dyn_state)

    # Calculate loss
    loss_v_x = final_state.v_x - next_state.v_x
    loss_v_y = final_state.v_y - next_state.v_y
    loss_r = final_state.r - next_state.r

    return loss_v_x + loss_v_y + loss_r
