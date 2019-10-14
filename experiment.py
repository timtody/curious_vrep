import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after):
    logger = Logger("local/test")
    agent = DQNAgent()
    trainer = Trainer(agent)
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)

    state = env.reset()
    for step in range(n_episodes):
        action = agent.get_action(state)
        next_state, reward, done, info =\
            env.step(agent.transform_action(action))
        agent.store_experience(state, next_state, action, reward)
        state = next_state

        if step % train_after == (train_after - 1):
            metrics_dict = agent.train()
            logger.log_metrics(metrics_dict, step)
