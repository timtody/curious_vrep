import os
import numpy as np
import tensorflow as tf
from imageio import get_writer
from skimage.color import rgb2gray


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        self._make_dir(logdir)
        self.tb_logger = TBLogger(logdir)
        self.vid_logger = VideoLogger(logdir)
        self.metrics_logger = MetricsLogger(logdir)

    def _make_dir(self, logdir):
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    def log_metrics(self, metrics_dict, step):
        self.tb_logger.log_metrics(metrics_dict, step)

    def log_video(self, frames, step):
        video_name = self._get_vid_name(step)
        self.vid_logger.make_video(frames, video_name)

    def log_video_with_debug_cams(self, vis, deb0, deb1, step):
        name = self._get_vid_name(step)
        self.vid_logger.make_video_with_debug_cams(vis, deb0, deb1, name)

    def _get_vid_name(self, step):
        path = os.path.join(self.logdir, "vid", f"frame{step}.mp4")
        return path


class MetricsLogger:
    def __init__(self, logdir):
        self._make_dir(logdir)

    def _make_dir(self, logdir):
        path = os.path.join(logdir, plots)
        if not os.path.exists(path):
            os.mkdir(path)

    def log_metrics(self, metrics_dict, step):
        for agent_name, m_dict in metrics_dict.items():
            for metric, value in m_dict.items():
                self._log_value(metric, value, step)

    def _log_value(self, key, value, step):
        pass


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

    @tf.function
    def log_metrics(self, metrics_dict, step):
        for agent_name, m_dict in metrics_dict.items():
            for key, value in m_dict.items():
                with self.writers[agent_name].as_default():
                    tf.summary.scalar(key, value, step=step)

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











