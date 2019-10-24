import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after, video_after,
            video_len, logdir=None):
    logger = Logger(logdir)
    agent = DQNAgent()
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)
    trainer = Trainer(env, agent)

    n_training_steps = n_episodes // train_after

    global_step = 0
    for step in range(n_episodes):
        print(f"episonde {step}")
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            n_state, reward, done, inf = env.step(action)
            agent.store_experience(state, n_state, action, reward)
            state = n_state

            if global_step % train_after == (train_after - 1):
                print("Training agents")
                metrics_dict = agent.train()
                logger.log_metrics(metrics_dict, global_step)
                agent.decrease_eps(n_training_steps)
                print(f"agent eps: {agent.joint_agents[0].eps}")

            if global_step % video_after == 0:
                print("logging video")
                vis, debug0, debug1 = trainer.record_frames(video_len, debug_cams=True)
                logger.log_vid_debug_cams(vis, debug0, debug1, global_step)

            global_step += 1
