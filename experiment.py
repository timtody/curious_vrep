import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes):
    agent = DQNAgent()
    trainer = Trainer(agent)
    env = Env(env_file, vision_handle)

    state = env.reset()
    for _ in range(n_episodes):
        action = agent.get_action(state)
        next_state = env.step(action)
        # subject to change
        reward = 0
        agent.store_experience(state, next_state, action, reward)
        state = next_state
