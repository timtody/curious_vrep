class Transition:
    def __init__(self):
        self.state_old = None
        self.state_new = None
        self.action = None
        self.reward = None

    def set_state_old(self, state_old):
        self.state_old = state_old

    def set_state_new(self, state_new):
        self.state_new = state_new

    def set_action(self, action):
        self.action = action

    def set_reward(self, reward):
        self.reward = reward
