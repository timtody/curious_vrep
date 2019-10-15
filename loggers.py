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
        video_name = self._get_vid_name(step)
        self.vid_logger.make_video(frames, video_name)

    def _get_vid_name(self):
        # todo: implement
        return "name"


class TBLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)

    def log_metrics(self, metrics_dict, step):
        with self.writer.as_default():
            for key, value in metrics_dict.items():
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
