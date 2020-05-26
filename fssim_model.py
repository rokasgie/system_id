import torch
from dataclasses import dataclass
from utils import tensor
import torch.nn.functional as F
import numpy as np
# from typing import List


@dataclass()
class State:
    x: torch.tensor
    y: torch.tensor
    r: torch.tensor
    yaw: torch.tensor

    v_x: torch.tensor
    v_y: torch.tensor
    a_x: torch.tensor = tensor(0)
    a_y: torch.tensor = tensor(0)

    @staticmethod
    def create(x, y, r, yaw, v_x, v_y):
        return State(tensor(x), tensor(y), tensor(r),
                     tensor(yaw), tensor(v_x), tensor(v_y))

    def __add__(self, other):
        return State(x=self.x + other.x, y=self.y + other.y,
                     v_x=self.v_x + other.v_x, v_y=self.v_y + other.v_y,
                     r=self.r + other.r, yaw=self.yaw + other.yaw)

    def __mul__(self, val):
        return State(x=self.x * val, y=self.y * val,
                     v_x=self.v_x * val, v_y=self.v_y * val,
                     r=self.r * val, yaw=self.yaw * val)


@dataclass()
class Controls:
    dc: torch.tensor
    delta: torch.tensor

    @staticmethod
    def create(delta, dc):
        return Controls(tensor(dc), tensor(delta))


@dataclass()
class Constants:
    INERTIA_G_: torch.tensor = tensor(9.81)
    INERTIA_M_: torch.tensor = tensor(190)
    INERTIA_I_Z_: torch.tensor = tensor(110)
    AERO_C_DOWN_: torch.tensor = tensor(1.9032)
    AERO_C_DRAG_: torch.tensor = tensor(0.7)
    KINEMATIC_L_: torch.tensor = tensor(1.53)
    KINEMATIC_L_F_: torch.tensor = tensor(1.22)
    KINEMATIC_L_R_: torch.tensor = tensor(1.22)
    KINEMATIC_W_FRONT_: torch.tensor = tensor(0.5)
    FRONT_AXLE_WIDTH_: torch.tensor = tensor(1)
    REAR_AXLE_WIDTH_: torch.tensor = tensor(1)
    TIRE_B_: torch.tensor = tensor(12.56)
    TIRE_C_: torch.tensor = tensor(-1.38)
    TIRE_D_: torch.tensor = tensor(1.6)
    TIRE_E_: torch.tensor = tensor(-0.58)
    DRIVETRAIN_CM1_: torch.tensor = tensor(5000)
    DRIVETRAIN_CR0_: torch.tensor = tensor(180)
    DRIVETRAIN_M_LON_: torch.tensor = tensor(29.98)

    # INERTIA_G_: torch.tensor = tensor(0.01)
    # INERTIA_M_: torch.tensor = tensor(0.01)
    # INERTIA_I_Z_: torch.tensor = tensor(0.01)
    # AERO_C_DOWN_: torch.tensor = tensor(0.01)
    # AERO_C_DRAG_: torch.tensor = tensor(0.01)
    # KINEMATIC_L_: torch.tensor = tensor(0.01)
    # KINEMATIC_L_F_: torch.tensor = tensor(0.01)
    # KINEMATIC_L_R_: torch.tensor = tensor(0.01)
    # KINEMATIC_W_FRONT_: torch.tensor = tensor(0.01)
    # FRONT_AXLE_WIDTH_: torch.tensor = tensor(0.01)
    # REAR_AXLE_WIDTH_: torch.tensor = tensor(0.01)
    # TIRE_B_: torch.tensor = tensor(0.01)
    # TIRE_C_: torch.tensor = tensor(0.01)
    # TIRE_D_: torch.tensor = tensor(0.01)
    # TIRE_E_: torch.tensor = tensor(0.01)
    # DRIVETRAIN_CM1_: torch.tensor = tensor(0.01)
    # DRIVETRAIN_CR0_: torch.tensor = tensor(0.01)
    # DRIVETRAIN_M_LON_: torch.tensor = tensor(0.01)

    def as_list(self):
        return [self.INERTIA_G_, self.INERTIA_M_, self.INERTIA_I_Z_,
                self.AERO_C_DOWN_, self.AERO_C_DRAG_,
                self.KINEMATIC_L_, self.KINEMATIC_L_F_, self.KINEMATIC_L_R_, self.KINEMATIC_W_FRONT_,
                self.FRONT_AXLE_WIDTH_, self.REAR_AXLE_WIDTH_,
                self.TIRE_B_, self.TIRE_C_, self.TIRE_D_, self.TIRE_E_,
                self.DRIVETRAIN_CR0_, self.DRIVETRAIN_CM1_, self.DRIVETRAIN_M_LON_]


@dataclass()
class Timestep:
    def __init__(self, prev_state: State, next_state: State, controls: Controls, dt):
        self.controls: Controls = controls
        self.prev_state: State = prev_state
        self.next_state: State = next_state
        # self.constants: Constants = constants
        self.dt = dt


class TimestepModel(torch.nn.Module):
    def __init__(self, constants):
        super(TimestepModel, self).__init__()
        self.constants: Constants = constants

    def getFyF(self, timestep: Timestep):
        return self.getFy(timestep, self.getFz(timestep), True)

    def getFyR(self, timestep: Timestep):
        return self.getFy(timestep, self.getFz(timestep), False)

    def getSlipAngle(self, timestep: Timestep, isFront=True):
        lever_arm_len = self.constants.KINEMATIC_L_ * self.constants.KINEMATIC_W_FRONT_
        v_x = torch.max(tensor(1), timestep.prev_state.v_x)
        if isFront:
            return torch.atan((timestep.prev_state.v_y + lever_arm_len * timestep.prev_state.r) /
                              (v_x - 0.5 + self.constants.FRONT_AXLE_WIDTH_ * timestep.prev_state.r)) \
                   - timestep.controls.delta
        else:
            return torch.atan((timestep.prev_state.v_y - lever_arm_len * timestep.prev_state.r) /
                              (v_x - 0.5 + self.constants.REAR_AXLE_WIDTH_ * timestep.prev_state.r))

    def getDownForceFront(self, Fz):
        return 0.5 * self.constants.KINEMATIC_W_FRONT_ * Fz

    def getFy(self, timestep: Timestep, Fz, isFront=True):
        slipAngle = self.getSlipAngle(timestep, isFront)
        Fz_axle = self.getDownForceFront(Fz)
        mu = self.constants.TIRE_D_ * torch.sin(self.constants.TIRE_C_ * torch.atan(self.constants.TIRE_B_ *
            (tensor(1) - self.constants.TIRE_E_) * slipAngle + self.constants.TIRE_E_ * torch.atan(self.constants.TIRE_B_ * slipAngle)))
        return Fz_axle * mu

    def getFDown(self, timestep: Timestep):
        return self.constants.AERO_C_DOWN_ * timestep.prev_state.v_x ** 2

    def getFz(self, timestep: Timestep):
        return self.constants.INERTIA_G_ * self.constants.INERTIA_M_ + self.getFDown(timestep)

    def getFdrag(self, timestep: Timestep):
        return self.constants.AERO_C_DRAG_ * timestep.prev_state.v_x ** 2

    def getFx(self, timestep: Timestep):
        # dc = tensor(np.arrange(timestep.controls.dc.size(0)))
        # for i in range(timestep.controls.dc.size(0)):
            # dc[i] = 0 if timestep.prev_state.v_x[i] <= 0 and timestep.controls.dc[i] < 0 else timestep.controls.dc[i]

        dc = timestep.controls.dc
        # dc = 0 if timestep.prev_state.v_x <= 0 and timestep.controls.dc < 0 else timestep.controls.dc
        return dc * self.constants.DRIVETRAIN_CM1_ - self.getFdrag(timestep) - self.constants.DRIVETRAIN_CR0_

    def kinCorrection(self, timestep: Timestep, next_dyn_state:State):
        v_x_dot = self.getFx(timestep) / (self.constants.INERTIA_M_ + self.constants.DRIVETRAIN_M_LON_)
        v = torch.sqrt(timestep.prev_state.v_x ** 2 + timestep.prev_state.v_y ** 2)
        v_blend = 0.5 * (v - 1.5)
        blend = torch.max(torch.min(tensor(1), v_blend), tensor(0))

        next_dyn_state.v_x = blend * next_dyn_state.v_x + (tensor(1) - blend) * (timestep.prev_state.v_x + timestep.dt * v_x_dot)

        v_y = torch.tan(timestep.controls.delta) * next_dyn_state.v_x * self.constants.KINEMATIC_L_R_ / self.constants.KINEMATIC_L_
        r = torch.tan(timestep.controls.delta) * next_dyn_state.v_x / self.constants.KINEMATIC_L_

        next_dyn_state.v_y = blend * next_dyn_state.v_y + (tensor(1) - blend) * v_y
        next_dyn_state.r = blend * next_dyn_state.r + (tensor(1) - blend) * r

        return next_dyn_state

    def forward(self, timestep: Timestep, criterion):
        x_dot = torch.cos(timestep.prev_state.yaw) * timestep.prev_state.v_x - \
                torch.sin(timestep.prev_state.yaw) * timestep.prev_state.v_y
        y_dot = torch.sin(timestep.prev_state.yaw) * timestep.prev_state.v_x + \
                torch.cos(timestep.prev_state.yaw) * timestep.prev_state.v_y

        yaw_dot = timestep.prev_state.r

        v_x_dot = (timestep.prev_state.r - timestep.prev_state.v_y) - \
                  (self.getFx(timestep) - torch.sin(timestep.prev_state.yaw) * 2 * self.getFyF(timestep)) / \
                  (self.constants.INERTIA_M_ + self.constants.DRIVETRAIN_M_LON_)

        v_y_dot = (torch.cos(timestep.controls.delta) * 2 * self.getFyF(timestep) + 2 * self.getFyR(timestep)) / \
                  self.constants.INERTIA_M_ - timestep.prev_state.r * timestep.prev_state.v_x

        r_dot = ((torch.cos(timestep.controls.delta) * 2 * self.getFyF(timestep) * self.constants.KINEMATIC_L_F_) -
                 (2 * self.getFyR(timestep) * self.constants.KINEMATIC_L_R_)) / self.constants.INERTIA_I_Z_

        dyn_state = State(x_dot, y_dot, yaw_dot, r_dot, v_x_dot, v_y_dot)
        next_dyn_state = timestep.prev_state + dyn_state * timestep.dt
        final_state = self.kinCorrection(timestep, next_dyn_state)

        # [F.relu(c) for c in self.constants.as_list()]

        # Calculate loss
        loss_x = criterion(final_state.x, timestep.next_state.x)
        loss_y = criterion(final_state.y, timestep.next_state.y)
        loss_v_x = criterion(final_state.v_x, timestep.next_state.v_x)
        loss_v_y = criterion(final_state.v_y, timestep.next_state.v_y)
        loss_yaw = criterion(final_state.yaw, timestep.next_state.yaw)
        loss_r = criterion(final_state.r, timestep.next_state.r)

        return loss_x + loss_y + loss_v_x + loss_v_y + loss_r + loss_yaw

