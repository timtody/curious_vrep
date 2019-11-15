import os
import hydra
import numpy as np
from matplotlib import pyplot as plt
from trainer import Trainer
from agents import DQNAgent
from environment import Env
from loggers import Logger


@hydra.main(config_path="configs/config.yaml")
def run_exp(cfg=None):
    logger = Logger(cfg)
    agent = DQNAgent(cfg)
    env = Env(cfg)
    trainer = Trainer(env, agent, cfg)

    cfg = cfg.exp
    n_training_steps = cfg.n_episodes // cfg.train_after
    global_step = 0
    state = env.reset()
    joint_angles = np.empty(cfg.n_episodes)
    for step in range(cfg.n_episodes):
        state = trainer.single_step(state)
        # agent training
        if global_step % cfg.train_after == (cfg.train_after - 1):
            print(f"step: {step}")
            print("Training agents")
            # fw model warmup phase of 2000 steps
            metrics_dict = agent.train(cfg.train_iv, cfg.train_fw,
                                       cfg.train_policy if global_step >= 2000 else False)
            logger.log_metrics(metrics_dict, global_step)
            logger.log_all_network_weights(agent.joint_agents[0], step)
            agent.decrease_eps(n_training_steps)

        # video logging
        if global_step % cfg.video_after == 0:
            print("logging video")
            vis, debug0, debug1 = trainer.record_frames(debug_cams=True)
            logger.log_vid_debug_cams(vis, debug0, debug1, global_step)

        # distractor toggling
        if global_step % cfg.toggle_table_after == (cfg.toggle_table_after - 1):
            env.toggle_table()

        global_step += 1
        pos = env.get_joint_positions()[0]
        joint_angles[step] = pos

    joint_angles = np.degrees(-joint_angles)
    plt.hist(joint_angles, bins=50, range=(0, 170))
    plt.savefig(os.path.join("plots", "explored_angles.png"))

if __name__ == "__main__":
    run_exp()
