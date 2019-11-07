import gin
import tensorflow as tf
import numpy as np
from models import dqn_model, ICModule
from replaybuffer import Buffer
from collections import defaultdict


@gin.configurable
class DQNAgent:
    """DQNAgent with intrinsic curiosity"""
    def __init__(self, n_discrete_actions, obs_shape, max_buffer_size, vel_min,
                 vel_max):
        self.buffer = Buffer(obs_shape, max_buffer_size=max_buffer_size,
                             n_agents=1)
        self._setup_joint_agents(n_discrete_actions)
        # actions transformed for the env
        self.env_actions = np.linspace(vel_min, vel_max, n_discrete_actions)

    def store_experience(self, transition):
        state = transition.state_old
        next_state = transition.state_new
        action = transition.action
        reward = transition.reward
        
        # do state processing such as convert to greyscale here
        # todo: change buffer.append() signature to append(transition) 
        self.buffer.append(state, next_state, action, reward)

    def get_action(self, obs):
        action = []
        for agent in self.joint_agents:
            action.append(agent.get_action(obs))

        return action

    def transform_action(self, action):
        return self.env_actions[action]

    def _setup_joint_agents(self, n_discrete_actions):
        self.joint_agents = []
        for i in range(1):
            self.joint_agents.append(JointAgent(self.buffer, n_discrete_actions, index=i))

    def train(self, train_iv, train_fw, train_policy):
        metrics_dict = {}
        for i, agent in enumerate(self.joint_agents):
            metrics_dict[i] = agent.train(train_iv=train_iv, train_fw=train_fw,
                                          train_policy=train_policy)
        metrics_dict = self._invert_metrics_dict(metrics_dict)

        return metrics_dict

    def _invert_metrics_dict(self, input_dict):
        """Is used to change the dict from Agent->metric->value to
        Metric->agent->value"""
        inverted_dict = defaultdict(dict)
        for agent_key, m_dict in input_dict.items():
            for metric, value in m_dict.items():
                inverted_dict[metric].update({agent_key: value})

        return inverted_dict

    def decrease_eps(self, n_training_steps):
        for agent in self.joint_agents:
            agent.decrease_eps(n_training_steps)


@gin.configurable
class JointAgent:
    def __init__(self, buffer, n_discrete_actions, start_eps, target_eps,
                 bsize, alph, index=None):
        self.policy = dqn_model()
        self.fw_model, self.iv_model, self.embed = ICModule().compile()
        self.buffer = buffer
        self.eps = start_eps
        self.start_eps = start_eps
        self.target_eps = target_eps
        self.bsize = bsize
        self.alph = alph
        self.index = index
        self.possible_actions = self._gen_actions(n_discrete_actions)

    def _gen_actions(self, n_actions):
        return np.arange(n_actions)

    def decrease_eps(self, n_training_steps):
        if self.eps >= self.target_eps:
            self.eps -= (self.start_eps - self.target_eps) / (n_training_steps
                                                              * 0.9)

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        draw = np.random.uniform()
        if draw <= self.eps:
            action = np.random.choice(self.possible_actions)
            return action
        predictions = self.policy.predict_on_batch(obs)
        predictions = np.squeeze(predictions)
        action = np.random.choice(self.possible_actions, p=predictions)

        return action

    def get_action_(self, obs):
        """Samples from the policy distribution"""
        obs = np.expand_dims(obs, axis=0)
        probs = self.policy.predict_on_batch(obs)

        return np.random.choice(self.possible_actions, p=probs)

    def train(self, train_iv=True, train_fw=True, train_policy=True):
        metrics_dict = {}
        trans = self._sample()
        # train inverse model
        if train_iv:
            metrics_dict.update(self._train_iv_model(trans))
        # train forward model
        if train_fw:
            m_dict, fw_loss = self._train_fw_model(trans)
            metrics_dict.update(m_dict)
        # train policy
        if train_policy:
            metrics_dict.update(self._train_policy(trans))

        return metrics_dict

    def _sample(self):
        old_states, new_states, actions, rewards =\
            self.buffer.get_random_batch(self.bsize)
        transition = {"old": old_states, "new": new_states,
                "actions": actions[:,self.index], "rewards": rewards}

        return transition

    def _train_policy(self, trans):
        pred_rewards_this = self.policy.predict_on_batch(trans["old"])
        pred_rewards_next = self.policy.predict_on_batch(trans["new"])
        target_rewards = trans["rewards"] +\
            self.alph * np.max(pred_rewards_next)
        network_targets = pred_rewards_this.numpy()

        # set the target rewards depending on the actual rewards
        for i in range(len(trans["actions"])):
            network_targets[i, int(trans["actions"][i])] =\
                target_rewards[i]
        history = self.policy.train_on_batch(trans["old"], network_targets)
        metrics_dict = {"policy_loss": history}

        return metrics_dict

    def _train_fw_model(self, trans):
        target_embedding = self.embed.predict_on_batch(trans["new"])
        loss = self.fw_model.fit(
            [trans["old"], np.expand_dims(trans["actions"], axis=-1)], target_embedding)
        metrics_dict = {"fw_model_loss": np.mean(loss)}

        return metrics_dict, loss

    def _train_iv_model(self, trans):
        loss, acc = self.iv_model.train_on_batch(
            [trans["old"], trans["new"]], trans["actions"])
        metrics_dict = {"iv_model_loss": loss, "iv_model_acc": acc}

        return metrics_dict
