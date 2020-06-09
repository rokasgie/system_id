import numpy as np
from lmfit import Parameters, Parameter
from dataclasses import dataclass


# Real parameters
# inertia:
#   m:        190.0   # Weight of the Vehicle [kg]
#   g:        9.81    # Gravity force         [m/s^2]
#   I_z:      110     # Inertial force I_zz

# kinematics:
#   l: 1.53           # Vehicle Length [m]
#   b_F: 1.22         # From COG to front axle [m]
#   b_R: 1.22         # From COG to rear axle [m]
#   w_front: 0.5      # Percentage of weight front

# tire:
#   tire_coefficient: 1.0
#   B: 12.56
#   C: -1.38
#   D: 1.60
#   E: -0.58
#   radius: 0.2525
#   max_steering: 0.8727

# aero:
#   C_Down: # F_Downforce = C_Downforce*v_x^2; C_Downforce = a*b*c
#     a: 1.22
#     b: 2.6
#     c: 0.6
#   C_drag: # F_Drag = C_Drag*v_x^2; C_drag = a*b*c
#     a: 0.7
#     b: 1.0
#     c: 1.0

# drivetrain:
#   Cm1: 5000
#   Cr0: 180
#   inertia: 0.4 # wheel_plus_packaging
#   r_dyn: 0.231
#   nm_wheels: 4

# torque_vectoring:
#   K_FFW: 2.1
#   K_p: 500
#   shrinkage: 0.8
#   K_stability: 0.004



def getFDown(params: Parameters, data):
    return params['AERO_C_DOWN_'] * (data.v_x ** 2)


def getFz(params: Parameters, data):
    return params['INERTIA_G_'] * params['INERTIA_M_'] + getFDown(params, data)


def getSlipAngle(params: Parameters, data, isFront=True):
    lever_arm_len = params['KINEMATIC_L_'] * params['KINEMATIC_W_FRONT_']
    v_x = np.maximum(1, data.v_x)
    if isFront:
        return np.arctan2((data.v_y + lever_arm_len * data.r),
                       v_x) \
               - data.delta_cmd
    else:
        return np.arctan2((data.v_y - lever_arm_len * data.r),
                       v_x)


def getDownForceFront(params: Parameters, Fz):
    return 0.5 * params['KINEMATIC_W_FRONT_'] * Fz


def getFy(params: Parameters, data, isFront=True):
    Fz = getFz(params, data)
    slipAngle = getSlipAngle(params, data, isFront)
    Fz_axle = getDownForceFront(params, Fz)
    B = params['TIRE_B_']
    C = params['TIRE_C_']
    D = params['TIRE_D_']
    E = params['TIRE_E_']

    mu = D * np.sin(C * np.arctan(B * (1.0 - E) * slipAngle + E * np.arctan(B * slipAngle)))
    return Fz_axle * mu


def getFyF(params: Parameters, data):
    return getFy(params, data, True)


def getFyR(params: Parameters, data):
    return getFy(params, data, False)


def getFdrag(params: Parameters, data):
    return params['AERO_C_DRAG_'] * (data.v_x ** 2)


def getFx(params: Parameters, data):
    return data.acc_cmd * params['DRIVETRAIN_CM1_'] - getFdrag(params, data) - params['DRIVETRAIN_CR0_']


def model(params, data, data_dot):

    Fx = getFx(params, data)
    FyF_tot = 2*getFyF(params, data)
    FyR_tot = 2*getFyR(params, data)
    m_lon = params['INERTIA_M_'] + params['DRIVETRAIN_M_LON_']

    v_x_dot = (data.r * data.v_y) + (Fx - np.sin(data.yaw) * FyF_tot) / m_lon

    v_y_dot = (np.cos(data.delta_cmd) * FyF_tot + FyR_tot) / params['INERTIA_M_'] - data.r * data.v_x

    r_dot = (np.cos(data.delta_cmd) * FyF_tot * params['KINEMATIC_L_F_'] - FyR_tot * params['KINEMATIC_L_R_']) / params['INERTIA_I_Z_']

    # Calculate loss
    loss_v_x = v_x_dot - data_dot.v_x_dot.values
    loss_v_y = v_y_dot - data_dot.v_y_dot.values
    loss_r = r_dot - data_dot.r_dot.values

    return loss_v_x + loss_v_y + loss_r
