import numpy as np
import tensorflow as tf
from imageio import get_writer
from skimage.color import rgb2gray


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.tb_logger = TBLogger(logdir)
        self.vid_logger = VideoLogger(logdir)

    def log_metrics(self, metrics_dict, step):
        self.tb_logger.log_metrics(metrics_dict, step)

    def log_video(self, frames, step):
        video_name = self._get_vid_name()
        self.vid_logger.make_video(frames, video_name)

    def _get_vid_name(self):
        # todo: implement
        return "name.mp4"


class TBLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writers = self._create_writers(logdir, 7)

    def _create_writers(self, logdir, n_agents):
        writers = {}
        for i in range(n_agents):
            name = f"{logdir}_agent{i}"
            writers[i] = tf.summary.create_file_writer(name)

        return writers

    def log_metrics(self, metrics_dict, step):
        for agent_name, m_dict in metrics_dict.items():
            for key, value in m_dict.items():
                with self.writers[agent_name].as_default():
                    tf.summary.scalar(key, value, step=step)

    def _get_summary_name(self, agent, key):
        return f"{key}_{agent}"


class VideoLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def make_video(self, images, name):
        frames = self._process_frames(images)
        with get_writer(name) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _process_frames(self, frames):
        frames = self._to_greyscales(frames)
        frames = self._rotate_frames(frames)
        frames = self._scale_frames(frames)

        return frames

    def _rotate_frames(self, frames):
        frames = np.transpose(frames, axes=[0,2,1])

        return frames

    def _to_greyscales(self, frames):
        frames = rgb2gray(frames)

        return frames

    def _scale_frames(self, frames):
        frames = (frames * 255).astype(np.uint8)

        return frames












