import numpy as np
from matplotlib import pyplot as plt
import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger

def minmax(pre, post):

    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")

def save_debug_img(pre_train, post_train, global_step):
    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")
    print(f"posttrain: max -> {np.max(post_train)} min -> {np.min(post_train)}")
    plot, axes = plt.subplots(ncols=2)
    axes[0].imshow(pre_train.numpy().squeeze().reshape((28, 28)))
    axes[1].imshow(post_train.numpy().squeeze().reshape((28, 28)))
    path = f"/home/taylor/results/img/check_iv_step_{global_step}"
    plt.savefig(path)
    plt.close()


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after, video_after,
            video_len, logdir=None):
    logger = Logger(logdir)
    agent = DQNAgent()
    env = Env(env_path=env_file, vis_name=vision_handle, headless=True)
    trainer = Trainer(env, agent)

    n_training_steps = n_episodes // train_after

    global_step = 0
    jt_agent = agent.joint_agents[0]
    logger.log_network_weights(jt_agent.embed, 0)
    for step in range(n_episodes):
        print(f"episode {step}")
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            n_state, reward, done, inf = env.step(action)
            agent.store_experience(state, n_state, action, reward)
            state = n_state

            if global_step % train_after == (train_after - 1):
                print("Training agents")
                pre_train = jt_agent.embed.predict_on_batch(np.expand_dims(state, axis=0))
                metrics_dict = agent.train()
                logger.log_network_weights(jt_agent.embed, global_step)
                post_train= jt_agent.embed.predict_on_batch(np.expand_dims(state, axis=0))
                #save_debug_img(pre_train, post_train, global_step)
                logger.log_metrics(metrics_dict, global_step)
                agent.decrease_eps(n_training_steps)
                print(f"agent eps: {agent.joint_agents[0].eps}")

            if global_step % video_after == 0:
                print("logging video")
                vis, debug0, debug1 = trainer.record_frames(video_len, debug_cams=True)
                logger.log_vid_debug_cams(vis, debug0, debug1, global_step)

            global_step += 1
