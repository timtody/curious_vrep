import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes):
    logger = Logger("local/test")
    agent = DQNAgent()
    trainer = Trainer(agent)
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)

    state = env.reset()
    for step in range(n_episodes):
        print("running a frame...")
        action = agent.get_action(state)
        print(f"action: {action}")
        next_state, reward, done, info =\
            env.step(agent.transform_action(action))
        # subject to change
        agent.store_experience(state, next_state, action, reward)
        state = next_state

        if step % 50 == 0:
            print("########## TRAINING ##############")
            agent.train(tb_callback=logger.tb_callback)
