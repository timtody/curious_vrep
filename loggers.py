import tensorflow as tf

class Logger:
    def __init__(self, path):
        self.tb_callback = self._create_tb_callback(path)

    def _create_tb_callback(self, path):
        cb = tf.keras.callbacks.TensorBoard(log_dir=path)

        return cb



