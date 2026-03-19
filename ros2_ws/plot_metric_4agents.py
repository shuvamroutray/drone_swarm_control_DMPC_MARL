import numpy as np
import matplotlib.pyplot as plt

# ================================
# Drone Names
# ================================
drones = ["cf231", "cf232", "cf233", "cf234"]

colors = ["r", "g", "b", "m"]

plt.figure(figsize=(10,6))

# ================================
# TRACKING ERROR PLOT
# ================================
for i, drone in enumerate(drones):

    data = np.load(f"{drone}_metrics.npy")

    t = data[:,0]
    tracking_error = data[:,1]

    plt.plot(t, tracking_error, 
             color=colors[i], 
             label=f"{drone}")

plt.xlabel("Time (s)")
plt.ylabel("Tracking Error (m)")
plt.title("Tracking Error vs Time (4 Drones)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ================================
# INTER-DRONE DISTANCE PLOT
# ================================
plt.figure(figsize=(10,6))

for i, drone in enumerate(drones):

    data = np.load(f"{drone}_metrics.npy")

    t = data[:,0]
    min_dist = data[:,2]

    plt.plot(t, min_dist, 
             color=colors[i], 
             label=f"{drone}")

# Safety threshold
plt.axhline(0.4, linestyle='--', color='k', 
            label="Safety Distance (0.4m)")

plt.xlabel("Time (s)")
plt.ylabel("Minimum Inter-Drone Distance (m)")
plt.title("Inter-Drone Distance vs Time (4 Drones)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
