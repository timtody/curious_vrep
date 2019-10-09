from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor


class Env:
    def __init__(self, env_path, vis_name, headless=True):
        self._launch(env_path, headless)
        self._setup_robot()
        self._setup_vision(vis_name)

    def _setup_robot(self):
        self.robot = Panda()
        self.joint_init = self.robot.get_joint_positions()
        self.robot.set_control_loop_enabled(False)
        self.robot.set_motor_locked_at_zero_velocity(True)

    def _setup_vision(self, vis_name):
        self.vision = VisionSensor(vis_name)

    def _launch(self, path, headless):
        self.pr = PyRep()
        self.pr.launch(path, headless=headless)
        self.pr.start()

    def step(self, action):
        self.robot.set_joint_target_velocities(action)
        self.pr.step()
        rgb = self.vision.capture_rgb()

        # change to include more meaningful info
        return rgb, 0, False, {}

    def reset(self):
        self.robot.set_joint_positions(self.joint_init)
        rgb = self.vision.capture_rgb()

        return rgb


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    env = Env("/home/julius/projects/curious_vrep/envs/test.ttt",
            "Vision_sensor")
    for i in range(100):
        action = np.random.uniform(size=7)
        rgb = env.step(action)
        print(rgb.shape)
        print(rgb)
        plt.imshow(rgb)
        plt.show()
    print(rgb.shape)
    print("donde")




