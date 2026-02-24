# test_dmpc_control.py

from control.DMPCControl_v1 import DMPCControl
import numpy as np

# Instantiate a single drone controller
dmpc_controller = DMPCControl(
    drone_id=0,
    Np=10, 
    dt=0.0333, # 1/30 Hz
    target_pos=np.array([1, 1, 1]),
    max_acc=5.0,
    max_vel=2.0,
    d_safe=0.4,
    num_drones=2 
)

# Test input (Drone 0 state, Drone 1 dummy predicted trajectory)
initial_state = np.array([0, 0, 0.5, 0, 0, 0]) 
dummy_neighbor_traj = np.zeros(6 * (10 + 1)) # Full predicted trajectory for 1 neighbor
# (If N=2, you'd only pass one neighbor's trajectory here)

# Compute control (this triggers the solver)
u_opt = dmpc_controller.compute_control(initial_state, dummy_neighbor_traj)

print("Optimal Acceleration u_opt:", u_opt)