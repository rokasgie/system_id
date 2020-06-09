import csv
from typing import List
from fssim_model import Timestep, State, Controls, Constants


class TimestepDataset(Dataset):
    def __init__(self, args, constants):
        self.timesteps = self.get_timesteps(args.file_path, args.skip_first, args.no_samples, args.use_every, constants)

    def get_timesteps(self, csv_file, skip, no_samples, take_every, constants: Constants):
        timesteps = []
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

        for i in range(len(data[reader.fieldnames[0]])):
            prev_state = State.create(x=data['x'][i], y=data['y'][i],
                                      r=data['r'][i], yaw=data['yaw'][i],
                                      v_x=data['vx'][i], v_y=data['vy'][i])

            next_state = State.create(x=data['x_dot'][i], y=data['y_dot'][i],
                                      r=data['r_dot'][i], yaw=data['yaw_dot'][i],
                                      v_x=data['vx_dot'][i], v_y=data['vy_dot'][i])

            controls = Controls.create(delta=data['delta'][i], dc=data['dc'][i])

            timestep = Timestep(prev_state, next_state, controls, constants, float(data['dt'][i]))
            timesteps.append(timestep)

        return timesteps

    def __getitem__(self, item):
        return self.timesteps[item]

    def __len__(self):
        return len(self.timesteps)


# def collate_fn(timesteps: List[Timestep]):
#     prev_state = State(x=tensor([t.prev_state.x for t in timesteps]), y=tensor([t.prev_state.y for t in timesteps]),
#                        v_x=tensor([t.prev_state.v_x for t in timesteps]), v_y=tensor([t.prev_state.v_y for t in timesteps]),
#                        yaw=tensor([t.prev_state.yaw for t in timesteps]), r=tensor([t.prev_state.r for t in timesteps]))
#
#     next_state = State(x=tensor([t.next_state.x for t in timesteps]),
#                        y=tensor([t.next_state.y for t in timesteps]),
#                        v_x=tensor([t.next_state.v_x for t in timesteps]),
#                        v_y=tensor([t.next_state.v_y for t in timesteps]),
#                        yaw=tensor([t.next_state.yaw for t in timesteps]),
#                        r=tensor([t.next_state.r for t in timesteps]))
#
#     controls = Controls(delta=tensor([t.controls.delta for t in timesteps]),
#                         dc=tensor([t.controls.dc for t in timesteps]))
#
#     dt = tensor([t.dt for t in timesteps])
#
#     return Timestep(prev_state, next_state, controls, timesteps[0].constants, dt)




