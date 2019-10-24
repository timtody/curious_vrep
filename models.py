import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate
from tensorflow.keras.layers import MaxPool2D, Flatten, Reshape
from tensorflow.keras.backend import expand_dims
tf.get_logger().setLevel('INFO')
#tf.compat.v1.disable_eager_execution()


@gin.configurable
def dqn_model(n_discrete_actions, input_shape):
    model = Sequential([
        Conv2D(8, (3, 3), activation="relu", input_shape=input_shape),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Conv2D(16, (3, 3), activation="relu"),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128),
        Dense(128),
        Dense(n_discrete_actions)
    ])
    model.compile(optimizer="adam", loss="mse")

    return model


@gin.configurable
class ICModule:
    def __init__(self, input_shape, n_discrete_actions):
        self.state_t0 = Input(shape=input_shape)
        self.state_t1 = Input(shape=input_shape)
        self.state_forward = Input(shape=input_shape)
        self.state_embedding = Input(shape=input_shape)
        self.action = Input(shape=(1,))
        self.conv1 = Conv2D(1, (3, 3), strides=(1, 1), activation="relu")
        self.conv2 = Conv2D(16, (3, 3), strides=(1, 1), activation="relu")
        self.conv3 = Conv2D(16, (3, 3), strides=(1, 1), activation="relu")
        self.conv4 = Conv2D(1, (3, 3), strides=(2, 2), activation="sigmoid")
        self.flatten = Flatten()
        self.dense1 = Dense(128)
        self.dense2 = Dense(n_discrete_actions, activation="softmax")
        self.dense_fw_1 = Dense(1024)
        self.dense_fw_2 = Dense(784, activation='sigmoid')

    def _inverse_embedding(self, input):
        """This inverse embedding is used by the inverse model
        to compute a dense representation of the state. It's also
        predicted by the forward model in order to avoid predicting
        pixels. Since the inverse embedding is part of the forward
        prediction (i.e. state x state -> action) this should learn
        to ignore parts of the input space which do not correlate with
        the agents actions.

        Arguments:
            input {tensor} -- states which get passed into the inverse
            embedding layer.

        Returns:
            tensor -- shape = (289,) (todo: check dims)
         """
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        return x

    def _forward_prediction(self):
        """The forward prediction uses the embedding of the
        current state plus the current action to predict the
        embedding of the next state. By predicting the embedding,
        predictions in pixels space are circumvented. This makes
        1) the problem easier because the embedding space has dimensions
        of only 288 but also 2) uses the invariant embedding comupted
        in the inverse embedding, which should ignore inputs which are
        irrelevant to the agent.

        Returns:
            tensor -- shape=(289,)
        """
        state = self._inverse_embedding(self.state_forward)
        x = concatenate([self.action, state])
        x = self.dense_fw_1(x)
        x = self.dense_fw_2(x)
        x = self.flatten(x)

        return x

    def _inverse_prediction(self):
        """Maps state t0 and state t1 to an action using the
        inverse embedding. Here the submodule 'inverse_embedding'
        should learn a representation which only takes into account
        things the agent can "change" with his own actions and makes him
        1) robust to input noise and distractors 2) learning a feature space
        of "interactable" things.

        Returns:
            tensor -- shape = (9,)
        """
        embed_t0 = self._inverse_embedding(self.state_t0)
        embed_t1 = self._inverse_embedding(self.state_t1)
        x = concatenate([embed_t0, embed_t1])
        x = self.dense1(x)
        x = self.dense2(x)
        #x = self.flatten(x)

        return x

    def compile(self):
        """ICModule.compile() uses the tf.keras API to compile two
        modules (iv_model, emb_model). These are then accessible with the
        standard keras .fit() and .evaluate() methods. Since tf.keras does
        not allow for retrieval of loss on a per sample basis, the fw_model
        needs to be "compiled" manually. In order to achieve this, the tf.keras
        Model subclassing api is used in ForwardModel(Model). It's then passed
        to AuxModel which uses tf.eager mode currently to compute the "custom"
        loss.
        The loss on a per-sample basis is needed, since it's used to change the
        reward of the agent in hindsight. On every step, prediction error needs
        to be addded to the actual reward supplied by the environment. Refer to
        paper for details.
        todo: change to graph mode to improve performance. Note, that this requires
        changes in the Agent.py file because CURAgent calls .numpy() on the resulting
        tensor. This is only possible in eager mode and needs to be changed to .eval()
        in a session.

        Returns:
            tf.keras.Model, AuxModel, tf.keras.Model -- refer to AuxModel in this file.
        """
        # inverse model (predics: s_t x s_t+1 -> action)
        predicted_action = self._inverse_prediction()
        iv_model = Model(inputs=[self.state_t0, self.state_t1],
            outputs=predicted_action)
        iv_model.compile(optimizer="rmsprop",
                         loss="sparse_categorical_crossentropy",
                         metrics=["sparse_categorical_accuracy"])

        # forward model (predicts: s_t x a -> s_t+1)
        # trun off training for the embedded model when constructing
        predicted_state = self._forward_prediction()
        fw_model = Model(inputs=[self.state_forward, self.action],
            outputs=predicted_state)
        fw_model = AuxModel(fw_model, trainables=[self.dense_fw_1.variables,
            self.dense_fw_2.variables])

        # embedding model
        emb_model = Model(inputs=self.state_embedding,
            outputs=self._inverse_embedding(self.state_embedding))
        emb_model.compile(optimizer="adadelta", loss="mean_squared_error")

        return fw_model, iv_model, emb_model


class ForwardModel(Model):
    def __init__(self, model):
        super(ForwardModel, self).__init__()
        self.model = model

    def call(self, inputs):
        return self.model(inputs)


class AuxModel:
    def __init__(self, model, trainables):
        self.trainables = trainables[0] + trainables[1]
        self.model = ForwardModel(model)
        self.loss = lambda x, y: tf.reduce_mean(tf.square(tf.subtract(x, y)), axis=0)
        self.optimizer = tf.keras.optimizers.Adadelta()

    def fit(self, inputs, labels):
        return np.random.randn(len(inputs))

   # @tf.function
    def _fit(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainables)
        self.optimizer.apply_gradients(zip(gradients, self.trainables))

        return loss
