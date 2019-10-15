import tensorflow as tf
from imageio import get_writer


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
        self.writer = tf.summary.create_file_writer(logdir)

    def log_metrics(self, metrics_dict, step):
        with self.writer.as_default():
            for agent_name, m_dict in metrics_dict.items():
                for key, value in m_dict.items():
                    name = self._get_summary_name(agent_name, key)
                    tf.summary.scalar(name, value, step=step)

    def _get_summary_name(self, agent, key):
        return f"{key}_{agent}"


class VideoLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def make_video(self, images, name):
        with get_writer(name) as writer:
            for frame in images:
                writer.append(frame)
