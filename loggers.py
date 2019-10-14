import tensorflow as tf
from imageio import get_writer


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.tb_logger = TBLogger(logdir)
        self.vid_logger = VideoLogger(logdir)

    def log_metrics(self, metrics_dict, step):
        tb_logger.log_metrics(metrics_dict, step)


class TBLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)

    def log_metrics(self, metrics_dict, step):
        with self.writer.as_default():
            for key, value in metrics_dict:
                tf.summary.scalar(key, value, step=step)


class VideoLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def _get_vid_name(self):
        pass

    def make_video(self, images, name):
        with get_writer(name) as writer:
            for frame in images:
                writer.append(frame)
