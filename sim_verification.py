import torch
from torch.autograd import Variable
from torch.optim import SGD
from model import Model
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Simulation vehicle model verification")


def getNormalForce(v):
    return INERTIA_G_ * INERTIA_M_ + getFDown()


def getFDown(v):
    return AERO_C_DOWN_ * v ** 2


def getSlipAngle(v_x, v_y, r, delta, isFront=True):
    lever_arm_len = KINEMATIC_L_ * KINEMATIC_W_FRONT_
    v_x = max(1, v_x)
    if isFront:
        return np.atan((v_y + lever_arm_len * r) / (v_x - 0.5 + FRONT_AXLE_WIDTH_ * r)) - delta
    else:
        return np.atan((v_y - lever_arm_len * r) / (v_x - 0.5 + REAR_AXLE_WIDTH_ * r))


def getDownForceFront(Fz):
    return 0.5 * KINEMATIC_W_FRONT_ * Fz


def getFy(v_x, v_y, r, delta, Fz, front):
    slipAngle = getSlipAngle(v_x, v_y, r, delta, front)
    Fz_axle = getDownForceFront(Fz)
    mu = TIRE_D_ * np.sin(TIRE_C_ * np.atan(TIRE_B_ * (1 - TIRE_E_) * slipAngle +
                                            TIRE_E_ * np.atan(TIRE_B_ * slipAngle)))
    return Fz_axle * mu


def getFdrag(v_x):
    return AERO_C_DOWN_ * v_x ** 2


def getFx(v_x, dc):
    dc = 0 if v_x <= 0 and dc < 0 else dc
    return dc * DRIVETRAIN_CM1_ - getFdrag(v_x) - DRIVETRAIN_CR0_


def get_dynamic_state(yaw, r, v_x, v_y, Fx, mtv, FyF, FyR):
    x_dot = np.cos(yaw) * v_x - np.sin(yaw) * v_y
    y_dot = np.sin(yaw) * v_x + np.cos(yaw) * v_y
    yaw_dot = r
    v_x_dot = (r * v_y) + (Fx - np.sin(delta) * 2 * FyF) / (INERTIA_M_ + DRIVETRAIN_M_LON_)
    v_y_dot = (np.cos(delta) * 2 * FyF + 2 * FyR) / INERTIA_M_ - r * v_x
    r_dot = ((np.cos(delta) * 2 * FyF * KINEMATIC_L_F_) - (2 * FyR * KINEMATIC_L_R_)) / INERTIA_I_Z_
    a_x = 0
    a_y = 0
    return (x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, a_x, a_y)


def kin_correction(Fx):
    v_x_dot = Fx / (INERTIA_M_ + DRIVETRAIN_M_LON_)
    # v_blend = (np.sqrt(v_x**2 + v_y**2) - 1.5)/2

    blend = max(min(1, (np.sqrt(v_x ** 2 + v_y ** 2) - 1.5) / 2), 0)
    v_x = blend * v_x + (1 - blend) * (v_x_prev + dt * v_x_dot)

    v_y_dot = np.tan(delta) * v_x * KINEMATIC_L_R_ / KINEMATIC_L_
    r_dot = np.tan(delta) * v_x / KINEMATIC_L_

    # Mixed up v_y
    v_y = blend * v_y + (1 - blend) * v_y_dot
    r = blend * r + (1 - blend) * r_dot


# state = [None]
# inputs = [None]
# dt = None
#
# v_x, v_y, r, yaw, delta = state
# dc = inputs
#
# Fz = getNormalForce(v_x)
# FyF = getFy(v_x, v_y, r, delta, Fz, True)
# FyR = getFy(v_x, v_y, r, delta, Fz, False)
# Fx = getFx(v_x, dc)
# mtv = 0
#
# x_dot_dyn = get_dynamic_state(yaw, r, v_x, v_y, Fx, mtv, FyF, FyR)
# dyn_state = state + x_dot_dyn * dt
# state = kin_correction(dyn_state, state)

if __name__ == "__main__":
    #First equation
    x_values1 = [i for i in range(10)]
    x_train1 = np.array(x_values1, dtype=np.float32).reshape(-1, 1)
    x_train1 = torch.from_numpy(x_train1)

    y_values1 = [2 * i + 1 for i in x_values1]
    y_train1 = np.array(y_values1, dtype=np.float32).reshape(-1, 1)
    y_train1 = torch.from_numpy(y_train1)

    #Second equation
    x_values2 = [i for i in range(10)]
    x_train2 = np.array(x_values2, dtype=np.float32).reshape(-1, 1)
    x_train2 = torch.from_numpy(x_train2)

    y_values2 = [i/2 + 2 for i in x_values2]
    y_train2 = np.array(y_values2, dtype=np.float32).reshape(-1, 1)
    y_train2 = torch.from_numpy(y_train2)

    #Setup
    criterion = torch.nn.MSELoss()
    model = Model(1, 1).cuda()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(4000):
        optimizer.zero_grad()
        out1, out2 = model(x_train1.cuda(), x_train2.cuda())

        # out = model(x_train1.cuda(), x_train2.cuda())

        # out = out1 + out2
        # labels = y_train1.cuda() + y_train2.cuda()
        # labels = torch.cat((y_train1.cuda(), y_train2.cuda()))
        # loss = criterion(out, labels)

        # loss1 = criterion(out1, y_train1.cuda())
        # loss2 = criterion(out2, y_train2.cuda())
        # loss = loss1 + loss2

        out = torch.cat((out1, out2))
        labels = torch.cat((y_train1.cuda(), y_train2.cuda()))
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        # print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        # print(x_train2.T)
        # print(out2.T)
        # print(y_train2.T)

    print(out.data.T)

    # print(out1.data.T)
    # print(out2.data.T)
