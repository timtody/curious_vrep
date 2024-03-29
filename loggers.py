import os
import json
import numpy as np
import tensorflow as tf
from imageio import get_writer
from skimage.color import rgb2gray
from collections import defaultdict
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, cfg):
        # todo: this is a quick fix due to hydra specifics
        # since hydra already creates a logdir, this does not need
        # to be done here
        #self._make_dir(self.logdir)
        self.logdir = ""

        self.vid_logger = VideoLogger(self.logdir)
        self.metrics_logger = MetricsLogger(self.logdir)
        self.tb_logger = TBLogger(self.logdir)

    def _make_dir(self, logdir):
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    def log_metrics(self, metrics_dict, step):
        self.tb_logger.log_metrics(metrics_dict, step)

    def log_network_params(self, agent, iv=False, fw=False, embed=False,
                           policy=False):
        pass
    
    def log_all_network_weights(self, agent, step):
        for nw in agent.networks:
            self.log_network_weights(nw, step)

    def log_network_weights(self, network, step):
        self.tb_logger.log_network_weights(network, step)

    def log_video(self, frames, step):
        video_name = self._get_vid_name(step)
        self.vid_logger.make_video(frames, video_name)

    def log_vid_debug_cams(self, vis, deb0, deb1, step):
        name = self._get_vid_name(step)
        self.vid_logger.make_video_with_debug_cams(vis, deb0, deb1, name)

    def _get_vid_name(self, step):
        path = os.path.join(self.logdir, "vid", f"frame{step}.mp4")
        return path

    def make_plots(self):
        self.metrics_logger.make_plots()


class MetricsLogger:
    def __init__(self, logdir):
        self._make_dir(logdir)
        self.writer = self._init_writer(logdir)
        self.plotter = Plotter(logdir)

    def _make_dir(self, logdir):
        path = os.path.join(logdir, "plots")
        if not os.path.exists(path):
            os.mkdir(path)

    def log_metrics(self, metrics_dict, step):
        self.writer.log_metrics(metrics_dict, step)

    def make_plots(self):
        metrics_dict = self.writer.get_dict()
        self.plotter.plot(metrics_dict)

    def _init_writer(self, logdir):
        writer = FileWriter(logdir)

        return writer


class FileWriter:
    def __init__(self, path):
        self.path = os.path.join(path, "metrics")
        self.is_initialized = False

    def log_metrics(self, metrics_dict, step):
        if not self.is_initialized:
            with open(self.path, 'w+') as fp:
                json.dump(metrics_dict, fp)
            self.is_initialized = True
        else:
            with open(self.path, 'r+') as fp:
                old_dict = json.load(fp)
                self._merge_dicts(old_dict, metrics_dict)
                fp.seek(0)
                json.dump(old_dict, fp)

    def _merge_dicts(self, dict0, dict1):
        #dict0 = defaultdict(dict, dict0)
        for key, value in dict1.items():
            for subkey, subvalue in value.items():
                dict0[str(key)][subkey].append(subvalue[0])

    def get_dict(self):
        with open(self.path, 'r') as fp:
            metrics_dict = json.load(fp)

        return metrics_dict


class Plotter:
    def __init__(self, logdir):
        pass

    def plot(self, metrics_dict):
        fig, axes = plt.subplots(ncols=len(metrics_dict))
        for ax in axes:
            pass
        for i, (key, value) in enumerate(metrics_dict.items()):
            axes[i].plot(value)
        plt.show()


class TBLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self._make_dir(logdir)
        self.writers = self._create_writers(logdir, 7)

    def _make_dir(self, logdir):
        path = os.path.join(logdir, "tb")
        if not os.path.exists(path):
            os.mkdir(path)

    def _create_writers(self, logdir, n_agents):
        writers = {}
        for i in range(n_agents):
            name = os.path.join(logdir, "tb", f"agent{i}")
            print(f"crating logger at path {name}")
            writers[i] = tf.summary.create_file_writer(name)

        return writers

    def log_metrics(self, metrics_dict, step):
        for metric, a_dict in metrics_dict.items():
            for agent_name, value in a_dict.items():
                with self.writers[agent_name].as_default():
                    tf.summary.scalar(metric, value, step=step)

    def log_network_weights(self, network, step):
        with self.writers[0].as_default():
            for layer in network.layers:
                weights = layer.weights
                for weight in weights:
                    tf.summary.histogram(weight.name, weight, step=step)

    def _get_summary_name(self, agent, key):
        return f"{key}_{agent}"


class VideoLogger:
    def __init__(self, logdir):
        self._make_dir(logdir)

    def _make_dir(self, logdir):
        vid_path = os.path.join(logdir, "vid")
        if not os.path.exists(vid_path):
            os.mkdir(vid_path)

    def make_video(self, images, name):
        frames = self._process_frames(images)
        self._make_video(frames, name)

    def make_video_with_debug_cams(self, vision, debug0, debug1, name):
        vis = self._process_frames(vision)
        deb0 = self._process_frames(debug0)
        deb1 = self._process_frames(debug1)
        out = self._merge_vision_with_debug_frames(vis, deb0, deb1)
        self._make_video(out, name)

    def _process_frames(self, frames):
        frames = self._to_greyscales(frames)
        frames = self._rotate_frames(frames)
        frames = self._scale_frames(frames)

        return frames

    def _merge_vision_with_debug_frames(self, vision, debug0, debug1):
        n_frames = vision.shape[0]
        out = np.ones(shape=(n_frames, 208, 64), dtype=np.uint8)
        out[:,0:64,:] = debug0
        out[:,72:136,:] = vision
        out[:,144:,:] = debug1

        return np.transpose(out, axes=[0,2,1])

    def _make_video(self, frames, name):
        with get_writer(name) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _rotate_frames(self, frames):
        frames = np.transpose(frames, axes=[0,2,1])

        return frames

    def _to_greyscales(self, frames):
        frames = rgb2gray(frames)

        return frames

    def _scale_frames(self, frames):
        frames = (frames * 255).astype(np.uint8)

        return frames


if __name__ == "__main__":
    logger = Logger("testlog")
    m_dict = {0: {"reward": [100], "model_loss": [11]}, 1: {"reward": [0],
                                                        "model_loss": [1000]}}
    logger.log_metrics(m_dict, 0)
    logger.log_metrics(m_dict, 1)
    logger.log_metrics(m_dict, 2)
    logger.log_metrics(m_dict, 3)

    m_dict = logger.metrics_logger.writer.get_dict()
    logger.make_plots()






