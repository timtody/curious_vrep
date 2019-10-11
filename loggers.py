import tensorflow as tf

class Logger:
    def __init(self, path):
        self.tb_callback = self._create_tb_callback()

    def _create_tb_callback(self, path):
        cb = tf.keras.callbacks.TensorBoard(log_dir=path)

        return cb



