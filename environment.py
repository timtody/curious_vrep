import gin
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor

@gin.configurable
class Env:
    def __init__(self, n_discrete_actions, env_path=None,
                 vis_name=None, headless=True, debug_cam0=None, debug_cam1=None):
        self._launch(env_path, headless)
        self._setup_robot()
        self._setup_vision(vis_name)
        self._setup_actions(n_discrete_actions)
        self._setup_debug_cameras(debug_cam0, debug_cam1)

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

    def _setup_actions(self, n_discrete_actions):
        self.poss_actions = np.linspace(-1, 1, n_discrete_actions)

    def _convert_action(self, action):
        return self.poss_actions[action]

    def _setup_debug_cameras(self, name0, name1):
        self.vis_debug0 = VisionSensor(name0)
        self.vis_debug1 = VisionSensor(name1)

    def step(self, action):
        self.robot.set_joint_target_velocities(action)
        self.pr.step()
        rgb = self.vision.capture_rgb()

        # todo: change to include more meaningful info
        return rgb, 0, False, {}

    def reset(self):
        self.robot.set_joint_positions(self.joint_init)
        rgb = self.vision.capture_rgb()

        return rgb

    def get_debug_images(self):
        debug0 = self.vis_debug0.capture_rgb()
        debug1 = self.vis_debug1.capture_rgb()

        return debug0, debug1


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    env = Env(10, env_path="/home/julius/projects/curious_vrep/envs/test.ttt",
              vis_name="Vision_sensor")
    for i in range(100):
        action = np.random.uniform(size=7)
        rgb = env.step(action)[0]
        plt.imshow(np.transpose(rgb, axes=[1, 0, 2]))
        plt.savefig(f"local/img/img{i}")
    print(rgb.shape)
    print("donde")





