import numpy as np
from transition import Transition

class Trainer:
    def __init__(self, env, agent, cfg):
        self.env = env
        self.agent = agent
        self.state = env.reset()
        self.next_state = self.state
        self.video_len = cfg.trainer.video_len

    def record_frames(self, debug_cams=True):
        if debug_cams:
            return self._record_frames_with_debug_cams()
        return self._record_frames()

    def _record_frames(self):
        out = np.empty((self.video_len, 64, 64, 3))
        for i in range(self.video_len):
            out[i] = self.state
            self.step(store=False)

        return out

    def _record_frames_with_debug_cams(self):
        vision = np.empty((self.video_len, 64, 64, 3))
        debug0 = np.empty((self.video_len, 64, 64, 3))
        debug1 = np.empty((self.video_len, 64, 64, 3))

        for i in range(self.video_len):
            vision[i] = self.state
            debug_im0, debug_im1 = self.env.get_debug_images()
            debug0[i] = debug_im0
            debug1[i] = debug_im1
            self.step(store=False)

        return vision, debug0, debug1

    def _reduce_eps(self):
        pass

    def step(self, store=True):
        action = self.agent.get_action(self.state)
        self.next_state, reward, done, info = self.env.step(action)
        if store:
            self.agent.store_experience(self.state, self.next_state, action,
                                        reward)
        self.state = self.next_state

        if done:
            self.env.reset()

    def single_step(self, old_state):
        action = self.agent.get_action(old_state)
        new_state, reward, _done, _inf = self.env.step(action)
        transition = Transition()
        transition.set_state_new(new_state)
        transition.set_state_old(old_state)
        transition.set_reward(reward)
        transition.set_action(action)
        self.agent.store_experience(transition)
        
        return new_state