import numpy as np
from matplotlib import pyplot as plt
import gin
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger
from transition import Transition


@gin.configurable
def run_exp(env_file, vision_handle, n_episodes, train_after, video_after,
            video_len, train_iv, train_fw, train_policy, show_distractor_after,
            toggle_table_after, logdir=None):
    logger = Logger(logdir)
    agent = DQNAgent()
    env = Env(env_path=env_file, vis_name=vision_handle, headless=False)
    trainer = Trainer(env, agent)

    n_training_steps = n_episodes // train_after

    global_step = 0
    jt_agent = agent.joint_agents[0]
    logger.log_network_weights(jt_agent.embed, 0)
    for step in range(n_episodes):
        joint_angles = np.empty(n_episodes)

        print(f"episode {step}")
        state = env.reset()
        action = agent.get_action(state)
        n_state, reward, done, inf = env.step(action)
        transition = Transition()
        transition.set_state_new(n_state)
        transition.set_state_old(state)
        transition.set_reward(reward)
        transition.set_action(action)
        agent.store_experience(transition)

        state = n_state
        if global_step % train_after == (train_after - 1):
            print("Training agents")
            metrics_dict = agent.train(train_iv, train_fw, train_policy)
            logger.log_network_weights(jt_agent.embed, global_step)
            logger.log_network_weights(jt_agent.fw_model, global_step)
            logger.log_network_weights(jt_agent.iv_model, global_step)
            logger.log_network_weights(jt_agent.policy, global_step)
            logger.log_metrics(metrics_dict, global_step)
            agent.decrease_eps(n_training_steps)

        if global_step % video_after == 0:
            print("logging video")
            vis, debug0, debug1 = trainer.record_frames(video_len, debug_cams=True)
            logger.log_vid_debug_cams(vis, debug0, debug1, global_step)

        if global_step % toggle_table_after == (toggle_table_after - 1):
            env.toggle_table()

        global_step += 1
        joint_angles[step] = env.get_joint_positions()

    plt.hist(joint_angles)
    plt.savefig("/home/julius/projects/curious_vrep/local/dist.png")

def get_embedding_img(agent, state):
    img = agent.embed.predict_on_batch(np.expand_dims(state, axis=0))
    return img

def minmax(pre, post):
    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")

def save_debug_img(pre_train, post_train, global_step):
    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")
    print(f"posttrain: max -> {np.max(post_train)} min -> {np.min(post_train)}")
    plot, axes = plt.subplots(ncols=2)
    axes[0].imshow(pre_train.numpy().squeeze().reshape((28, 28)))
    axes[1].imshow(post_train.numpy().squeeze().reshape((28, 28)))
    path = f"/home/julius/results/img/check_iv_step_{global_step}"
    plt.savefig(path)
    plt.close()
