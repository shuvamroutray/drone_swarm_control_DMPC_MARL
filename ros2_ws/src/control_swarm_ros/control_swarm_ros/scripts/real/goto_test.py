#!/usr/bin/env python3

from crazyflie_py import Crazyswarm


def main():

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    print("Taking off...")
    cf.takeoff(targetHeight=0.6, duration=3.0)
    timeHelper.sleep(5.0)

    print("Moving +X (forward)...")
    cf.goTo([0.3, 0.0, 0.6], 0.0, 3.0)
    timeHelper.sleep(4.0)

    print("Returning to origin...")
    cf.goTo([0.0, 0.0, 0.6], 0.0, 3.0)
    timeHelper.sleep(4.0)

    print("Moving +Y (left)...")
    cf.goTo([0.0, 0.3, 0.6], 0.0, 3.0)
    timeHelper.sleep(4.0)

    print("Returning to origin...")
    cf.goTo([0.0, 0.0, 0.6], 0.0, 3.0)
    timeHelper.sleep(4.0)

    print("Landing...")
    cf.land(targetHeight=0.03, duration=1.5)
    timeHelper.sleep(3.0)

    print("Done.")


if __name__ == "__main__":
    main()