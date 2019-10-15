class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.state = env.reset()
        self.next_state = self.state

    def record_frames(self, frames):
        pass

    def _reduce_eps(self):
        pass

    def step(self):
        action = self.agent.get_action(self.state)
        self.next_state, reward, done, info =\
            self.env.step(self.agent.transform_action(action))
        self.agent.store_experience(self.state, self.next_state, action,
                                    reward)
        self.state = self.next_state

    def set_parameters(self):
        pass
