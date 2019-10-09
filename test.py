import numpy as np
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.panda import Panda
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkagg")

class Robot:
    def __init__(self):
        self.arm = Panda()
        self.vis = VisionSensor("Vision_sensor")
        self.arm.set_motor_locked_at_zero_velocity(True)
        self.arm.set_control_loop_enabled(False)

pr = PyRep()
pr.launch('/home/julius/projects/curious_vrep/envs/test.ttt', headless=True)
#pr.launch('/home/julius/.software/pyrep/examples/scene_reinforcement_learning_env.ttt')
pr.start()


agent = Robot()

for i in range(1000):
    print("setting target velocities")
    vels = np.random.uniform(-1, 1, size=7)
    agent.arm.set_joint_target_velocities(vels)
    pr.step()
    im = agent.vis.capture_rgb()
    plt.imshow(im)
    plt.savefig(f"local/im{i}.png")

print("done with loop")
pr.stop()
pr.shutdown()
