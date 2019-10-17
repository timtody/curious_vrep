import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after, video_after):
    logger = Logger("local/test")
    agent = DQNAgent()
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)
    trainer = Trainer(env, agent)

    for step in range(n_episodes):
        print("training a step")
        trainer.step()

        if step % train_after == (train_after - 1):
            metrics_dict = agent.train()
            logger.log_metrics(metrics_dict, step)

        if step % video_after == (video_after - 1):
            print("logging vid")
            frames = trainer.record_frames(30)
            logger.log_video(frames, step)

