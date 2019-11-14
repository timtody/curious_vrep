import numpy as np


class Trainer:
    def __init__(self, env, agent, cfg):
        self.env = env
        self.agent = agent
        self.state = env.reset()
        self.next_state = self.state
        self.video_len = cfg.video_len

    def record_frames(self, n_frames, debug_cams=True):
        if debug_cams:
            return self._record_frames_with_debug_cams(n_frames)
        return self._record_frames(n_frames)

    def _record_frames(self, frames):
        out = np.empty((frames, 64, 64, 3))
        for i in range(frames):
            out[i] = self.state
            self.step(store=False)

        return out

    def _record_frames_with_debug_cams(self, n_frames):
        vision = np.empty((n_frames, 64, 64, 3))
        debug0 = np.empty((n_frames, 64, 64, 3))
        debug1 = np.empty((n_frames, 64, 64, 3))

        for i in range(n_frames):
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

    def set_parameters(self):
        pass
