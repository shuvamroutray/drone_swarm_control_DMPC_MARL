import numpy as np
import matplotlib.pyplot as plt

# Load both drones
data1 = np.load("cf231_metrics.npy")
data2 = np.load("cf232_metrics.npy")

# Extract columns
t1 = data1[:, 0]
err1 = data1[:, 1]
inter1 = data1[:, 2]

t2 = data2[:, 0]
err2 = data2[:, 1]
inter2 = data2[:, 2]

# =====================================================
# 1️⃣ Tracking Error (Both Drones Same Graph)
# =====================================================
plt.figure(figsize=(8,5))

plt.plot(t1, err1, label="Drone cf231")
plt.plot(t2, err2, label="Drone cf232")

plt.xlabel("Time (s)")
plt.ylabel("Tracking Error (m)")
plt.title("Tracking Error vs Time (Mission Change Scenario)")
plt.legend()
plt.grid(True)

# =====================================================
# 2️⃣ Inter-Drone Distance
# (Only need one since distance is symmetric)
# =====================================================
plt.figure(figsize=(8,5))

plt.plot(t1, inter1, label="Inter-drone distance")
plt.axhline(0.4, linestyle='--', label="Safety threshold (0.4m)")

plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.title("Inter-Drone Distance vs Time")
plt.legend()
plt.grid(True)

plt.show()
