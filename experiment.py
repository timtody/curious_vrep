import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes):
    agent = DQNAgent()
    trainer = Trainer(agent)
    env = Env(env_path=env_file, vis_name=vision_handle, headless=False)

    state = env.reset()
    for step in range(n_episodes):
        print("running a frame...")
        action = agent.get_action(state)
        print(f"action: {action}")
        next_state, reward, done, info = env.step(action)
        # subject to change
        agent.store_experience(state, next_state, action, reward)
        state = next_state

        if step % 50 == 0:
            print("########## TRAINING ##############")
            agent.train()
