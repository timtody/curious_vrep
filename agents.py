import gin
import numpy as np
from models import dqn_model
from replaybuffer import Buffer

@gin.configurable
class DQNAgent:
    """DQNAgent with intrinsic curiosity"""
    def __init__(self, eps, alph, bsize,
            nactions, obs_shape, max_buffer_size):
        self.eps = eps
        self.alph = alph
        self.bsize = bsize
        self.nactions = nactions
        self.buffer = Buffer(obs_shape, max_buffer_size=max_buffer_size,
                nactions=nactions)
        self.actions = self._gen_actions(nactions)
        self._setup_models()

    def store_experience(self, state, next_state, action, reward):
        # do state processing such as convert to greyscale here
        self.buffer.append(state, next_state, action, reward)

    def get_action(self, obs):
        draws = np.random.uniform(size=7)
        action = np.empty(7)
        for i, dr in enumerate(draws):
            if dr <= self.eps:
                action[i] = np.random.choice(self.actions)
            else:
                action[i] = self._get_action(obs, i)

        return action

    def _predict_rewards(self, obs):
         pred_rewards = []
         for model in self.models:
             pred_rewards.append(model.predict(obs))

        return pred_rewards

    def train(self):
        # train inverse model

        # train forward model

        # train policy

    def _train_policy(self):
        old_s, new_s, actions, rewards =\
            self.buffer.get_random_batch(self.bsize)

    def _setup_models(self):
        self.models = []
        for i in range(self.nactions):
            self.models.append(dqn_model())

    def _get_action(self, obs, joint):
        """Gets the action for joint number joint. Seven joints in total for the
        panda agent."""
        action = np.argmax(self.models[i].predict(obs))

        return action

    def _gen_actions(self, n):
        return np.linspace(-1, 1, n)


class JointAgent:
    def __init__(self):
        self.policy = dqn_model()
        self.fw_model, self.iw_model, self.embed = ICModule()

