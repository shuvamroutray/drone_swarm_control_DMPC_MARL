#!/usr/bin/env python3
import numpy as np

ACTIONS = {
    0: (0, 1),
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0),
    4: (0, 0),
}


# def world_to_grid(x, y, scale, grid_size):
#     gx = int(x / scale)
#     gy = int(y / scale)
#     return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)

def world_to_grid(x, y, scale, grid_size):
    gx = int(round((x / scale) + 2.5))
    gy = int(round((y / scale) + 2.5))
    return np.clip(gx, 0, grid_size - 1), np.clip(gy, 0, grid_size - 1)







# def grid_to_world(gx, gy, scale):
#     return (gx + 0.5) * scale, (gy + 0.5) * scale

def grid_to_world(gx, gy, scale):

    world_x = (gx - 2.5) * scale
    world_y = (gy - 2.5) * scale

    return world_x, world_y


def build_observation(grid, positions, drone_id, grid_size):

    gx, gy = positions[drone_id]

    flat_grid = grid.flatten()

    # normalize own position
    own_pos = np.array([
        gx / grid_size,
        gy / grid_size
    ])

    # relative positions
    other_positions = []
    for i, (ox, oy) in sorted(positions.items()):
        if i == drone_id:
            continue

        other_positions.extend([
            (ox - gx) / grid_size,
            (oy - gy) / grid_size
        ])

    # one-hot id
    id_vec = np.zeros(len(positions))
    id_vec[drone_id] = 1

    obs = np.concatenate([
        flat_grid,
        own_pos,
        np.array(other_positions),
        id_vec
    ])

    return obs.astype(np.float32)