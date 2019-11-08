import gin
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

@gin.configurable
class Env:
    def __init__(self, n_discrete_actions, vel_min, vel_max, env_path=None,
                 vis_name=None, headless=True, debug_cam0=None,
                 debug_cam1=None, enables_joints=None):
        self.enables_joints = enables_joints
        self._launch(env_path, headless)
        self._setup_robot()
        self._setup_vision(vis_name)
        self._setup_actions(n_discrete_actions, vel_min, vel_max)
        self._setup_debug_cameras(debug_cam0, debug_cam1)
        self._setup_target()
        self._setup_distractor()

    def _setup_distractor(self):
        self.distractor = Shape('distractor')
        self.hide_distractor()

    def _setup_target(self):
        self.target = Shape('target')

    def _setup_robot(self):
        self.robot = Panda()
        self.joint_init = self.robot.get_joint_positions()
        self.robot.set_control_loop_enabled(False)
        self.robot.set_motor_locked_at_zero_velocity(True)
        self.tip = self.robot.get_tip()

    def _setup_vision(self, vis_name):
        self.vision = VisionSensor(vis_name)

    def _launch(self, path, headless):
        self.pr = PyRep()
        self.pr.launch(path, headless=headless)
        self.pr.start()

    def _setup_actions(self, n_discrete_actions, vel_min, vel_max):
        self.poss_actions = np.linspace(vel_min, vel_max, n_discrete_actions)

    def _convert_action(self, action):
        return self.poss_actions[action]

    def _setup_debug_cameras(self, name0, name1):
        self.vis_debug0 = VisionSensor(name0)
        self.vis_debug1 = VisionSensor(name1)

    def step(self, action):
        converted_action = self._convert_action(action)
        action = np.zeros(7)
        action[self.enables_joints] = converted_action
        self.robot.set_joint_target_velocities(action)
        self.pr.step()
        reward = self._calculate_reward()

        if reward >= -0.15:
            done = True
            reward = 1
        else:
            done = False
        rgb = self.vision.capture_rgb()

        # todo: change to include more meaningful info
        return rgb, reward, done, {}

    def show_distractor(self):
        self.distractor.set_position([0.15, 0., 1.])

    def hide_distractor(self):
        self.distractor.set_position([-1.5, 0., 1.])

    def _calculate_reward(self):
        ax, ay, az = self.tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

        return reward

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

    env = Env(3, -3, 3, env_path="/home/julius/projects/curious_vrep/envs/test.ttt",
              vis_name="Vision_sensor", debug_cam0="debug_vis1",
              debug_cam1="debug_vis2", headless=True)
    for i in range(100000):
        action = np.random.choice([0,1,2])
        action = [0]
        rgb, reward, done, info = env.step([action])
        plt.imshow(rgb)
        plt.show()
    print(rgb.shape)
    print("done")





