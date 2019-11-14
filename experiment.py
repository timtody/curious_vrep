import os
import hydra
import numpy as np
from matplotlib import pyplot as plt
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger
from transition import Transition


@hydra.main
def run_exp(cfg):
    logger = Logger(cfg.exp.logger)
    agent = DQNAgent(cfg.cfg.exp)
    env = Env(cfg.env)
    trainer = Trainer(env, agent, cfg.trainer)

    n_training_steps = n_episodes // train_after

    global_step = 0
    jt_agent = agent.joint_agents[0]
    logger.log_network_weights(jt_agent.embed, 0)
    state = env.reset()
    joint_angles = np.empty(cfg.exp.n_episodes)
    for step in range(cfg.exp.n_episodes):
        #print(f"episode {step}")
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

        if global_step % cfg.exp.video_after == 0:
            print("logging video")
            vis, debug0, debug1 = trainer.record_frames(debug_cams=True)
            logger.log_vid_debug_cams(vis, debug0, debug1, global_step)

        if global_step % toggle_table_after == (toggle_table_after - 1):
            env.toggle_table()

        global_step += 1
        # max value [-0.0696348]
        # min value [-3.07196569]
        pos = env.get_joint_positions()[0]
        joint_angles[step] = pos

    joint_angles = np.degrees(-joint_angles)
    plt.hist(joint_angles)
    plt.savefig(os.path.join(logdir, "plots", "explored_angles.png"))

def get_embedding_img(agent, state):
    img = agent.embed.predict_on_batch(np.expand_dims(state, axis=0))
    return img

def save_debug_img(pre_train, post_train, global_step):
    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")
    print(f"posttrain: max -> {np.max(post_train)} min -> {np.min(post_train)}")
    plot, axes = plt.subplots(ncols=2)
    axes[0].imshow(pre_train.numpy().squeeze().reshape((28, 28)))
    axes[1].imshow(post_train.numpy().squeeze().reshape((28, 28)))
    path = f"/home/julius/results/img/check_iv_step_{global_step}"
    plt.savefig(path)
    plt.close()
