import gin
import numpy as np
from models import dqn_model, ICModule
from replaybuffer import Buffer

@gin.configurable
class DQNAgent:
    """DQNAgent with intrinsic curiosity"""
    def __init__(self, nactions, obs_shape, max_buffer_size):
        self.buffer = Buffer(obs_shape, max_buffer_size=max_buffer_size,
                nactions=nactions)
        self._setup_joint_agents(nactions)

    def store_experience(self, state, next_state, action, reward):
        # do state processing such as convert to greyscale here
        self.buffer.append(state, next_state, action, reward)

    def get_action(self, obs):
        action = []
        for agent in self.joint_agents:
            action.append(agent.get_action(obs))

        return action

    def _setup_joint_agents(self, nactions):
        self.joint_agents = []
        for _ in range(7):
            self.joint_agents.append(JointAgent(self.buffer, nactions))

    def train(self):
        for agent in self.joint_agents:
            agent.train()


@gin.configurable
class JointAgent:
    def __init__(self, buffer, n_discrete_actions,
            eps, bsize, alph):
        self.policy = dqn_model()
        self.fw_model, self.iv_model, self.embed = ICModule().compile()
        self.buffer = buffer
        self.eps = eps
        self.bsize = bsize
        self.alph = alph
        self.possible_actions =  self._gen_actions(n_discrete_actions)

    def _gen_actions(self, n_actions):
        return np.linspace(-1, 1, n_actions)

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        draw = np.random.uniform()
        if draw <= self.eps:
            return np.random.choice(self.possible_actions)
        else:
            return np.argmax(self.policy.predict(obs))

    def train(self):
        trans = self._sample()
        # train inverse model
        iv_loss = self._train_iv_model(trans)
        # train forward model
        fw_loss = self._train_fw_model(trans)
        # train policy
        policy_loss = self._train_policy(trans)

    def _sample(self):
        old_states, new_states, actions, rewards =\
            self.buffer.get_random_batch(self.bsize)
        transition = {"old": old_states, "new": new_states,
                "actions": actions, "rewards": rewards}

        return transition

    def _train_policy(self, trans):
        pred_rewards_this = self.policy.predict(trans["old"])
        pred_rewards_next = self.policy.predict(trans["new"])
        target_rewards = trans["rewards"] + self.alph * np.max(pred_rewards_next)
        network_targets = pred_rewards_this

        # set the target rewards depending on the actual rewards
        for i in range(len(trans["actions"])):
            network_targets[i, int(trans["actions"])] =\
                target_rewards[i]
        history = self.policy.fit(trans["old"], network_targets)
        metrics_dict = {"policy_loss": history.history["loss"][0]}

        return metrics_dict

    def _train_fw_model(self, trans):
        loss = self.fw_model.fit(
            [trans["old"], np.expand_dims(trans["actions"],axis=-1)],
            self.embed.predict(trans["new"])
        )
        metrics_dict = {"fw_model_loss": loss}

        return metrics_dict

    def _train_iv_model(self, trans):
        history = self.iv_model.fit(
            [trans["old"], trans["new"]], trans["actions"])
        metrics_dict = {"iv_model_loss": history.history["loss"][0]}

        return metrics_dict









