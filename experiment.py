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
        print(f"running step {step}")
        trainer.step()

        if step % train_after == (train_after - 1):
            print("########################################")
            metrics_dict = agent.train()
            logger.log_metrics(metrics_dict, step)

<<<<<<< HEAD
        if step % video_after == 0:
=======
        if step % video_after == (video_after - 1):
            print("#########################################")
>>>>>>> 61d3a1c1fae31bf087d82d8200596d9f56f7f886
            vis, debug0, debug1 =\
                trainer.record_frames_with_debug_cams(video_len)
            logger.log_video_with_debug_cams(vis, debug0, debug1, step)

