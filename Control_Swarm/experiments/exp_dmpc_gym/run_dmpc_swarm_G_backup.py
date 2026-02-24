# run_dmpc_swarm.py

import time
import numpy as np
import pybullet as p
# Correct package imports
from gym_pybullet_drones.envs.DMPCAviary import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def run_dmpc_swarm():
    NUM_DRONES = 2
    # Set initial positions (e.g., side by side)
    INIT_XYZS = np.array([
        [0, -0.5, 1.0], 
        [0, 0.5, 1.0]
    ])
    
    # --- Instantiate the Environment ---
    env = DMPCAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB_GND_DRAG_DW, # Use realistic physics
        gui=True, # Keep GUI True for debugging
        record=False,
        target_pos=np.array([2.0, 0.0, 1.5]) # Target for all drones
    )

    # --- Run the Simulation Loop ---
    
    obs, info = env.reset()
    
    START = time.time()
    
    # Dummy action: The DMPCAviary._preprocessAction handles the control calculation.
    DUMMY_ACTION = np.zeros((NUM_DRONES, 1))

    # Run for 20 seconds
    for i in range(env.PYB_FREQ * 20): 
        
        # The step function calls _preprocessAction, which runs the DMPC loop
        obs, reward, terminated, truncated, info = env.step(DUMMY_ACTION)
        
        # Check for termination conditions
        if terminated or truncated:
            break
        
        if i % env.PYB_FREQ == 0:
            env.render()
            
    # --- Clean Up ---
    env.close()
    
    elapsed_time = time.time() - START
    print(f"\nSimulation ended in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    run_dmpc_swarm()