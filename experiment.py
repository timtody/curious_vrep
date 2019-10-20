import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after, video_after,
            logdir, video_len):
    logger = Logger(logdir)
    agent = DQNAgent()
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)
    trainer = Trainer(env, agent)

    for step in range(n_episodes):
        trainer.step()

        if step % train_after == (train_after - 1):
            metrics_dict = agent.train()
            logger.log_metrics(metrics_dict, step)

        if step % video_after == 0:
            vis, debug0, debug1 =\
                trainer.record_frames_with_debug_cams(video_len)
            logger.log_video_with_debug_cams(vis, debug0, debug1, step)

