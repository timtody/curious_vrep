import numpy as np
from transition import Transition

class Buffer:
    def __init__(self, observation_shape, max_buffer_size=500, alpha=2,
            n_agents=1):
        """Initializes buffer which stores a tuple of observations
           containing old_state, new_state, action, reward of arbitrary sizes

        Arguments:
            observation_shape {tuple} -- must be of shape (observation_dim0 x observation_dim1)
             i.e. just pass observation.shape here

        Keyword Arguments:
            max_buffer_size {int} -- maximum size of the buffer (default: {500})
        """
        self.alpha = alpha
        self.max_buffer_size = max_buffer_size
        self.old_state = np.empty(shape=(max_buffer_size, *observation_shape))
        self.new_state = np.empty(shape=(max_buffer_size, *observation_shape))
        self.action = np.empty(shape=(max_buffer_size, n_agents))
        self.reward = np.empty(max_buffer_size)
        self.read_idx = 0
        self.write_idx = 0
        self.empty = self._isempty
        self.full = self._isfull
        self.filled = False

    def append(self, old_state, new_state, action, reward):
        """Appends a single batch to the buffer. old_state and
           new_state must be of the size specified in the
           constructor.

        Arguments:
            old_state {array-like}
            new_state {array-like}
            reward {int}
            action {int}
        """
        if self.full():
            self.write_idx = 0
            self.filled = True
        # append values to buffers
        self.old_state[self.write_idx] = old_state
        self.new_state[self.write_idx] = new_state
        self.action[self.write_idx] = action
        self.reward[self.write_idx] = reward
        self.write_idx += 1

    def next_batch_deprecated(self, batch_size):
        # !!! deprecated
        # please do not use in current state
        """get a batch of size batch_size from the buffer if possible.
           return empty arrays otherwise

        Arguments:
            batch_size {int} -- determines the size of the resulting batch

        Raises:
            IndexError: in case I was too stupid to calculate remainder_size

        Returns:
            old_state -- batch of old states with size (batch_size x observation_dim0 x observation_dim1)
            new_state -- batch of old states with size (batch_size x observation_dim0 x observation_dim1)
            old_state -- batch of old states with size (batch_size)
            old_state -- batch of old states with size (batch_size)
        """
        # calculate currently possible batch size: at least zero
        # at most batch size but never more than write_idx - read_idx)
        remainder_size = min(batch_size, max(0, self.write_idx - self.read_idx))
        if remainder_size == 0:
            # setting empty to True is a duplicate
            old_state = np.array([])
            new_state = np.array([])
            action = np.array([])
            reward = np.array([])
        elif remainder_size > 0:
            old_state = self.old_state[self.read_idx:self.read_idx + remainder_size]
            new_state = self.new_state[self.read_idx:self.read_idx + remainder_size]
            action = self.action[self.read_idx:self.read_idx + remainder_size]
            reward = self.reward[self.read_idx:self.read_idx + remainder_size]
            self.read_idx += remainder_size

        return old_state, new_state, action, reward

    def shuffle(self, seed=False):
        """shuffles the current buffer but only the values which have already been written

        Keyword Arguments:
            seed {int} -- pass a seed for debugging purpose if need be (default: {False})
        """
        if seed: np.random.seed(seed)
        # size of content to shuffle is max_buffer_size if a wrap around has happened
        # else its write_idx because we dont want to shuffle empty values
        content_size = self.write_idx if not self.full() else self.max_buffer_size
        _permutation = np.random.permutation(content_size)
        # this exists only to confuse readers
        shuf = lambda x: np.random.shuffle(x[:content_size])
        map(shuf, [self.old_state, self.new_state, self.action, self.reward])

    def get_random_batch(self, batch_size):
        if not self.filled:
            indices = np.random.randint(self.write_idx, size=batch_size)
        else:
            indices = np.random.randint(self.max_buffer_size, size=batch_size)
        transition = Transition()
        transition.set_state_old(self.old_state[indices])
        transition.set_state_new(self.new_state[indices])
        transition.set_action(self.action[indices])
        transition.set_reward(self.reward[indices])

        return transition

    def get_weighted_batch(self, batch_size):
        """This function generates weights depending on the
        reward of a transition. The higher the reward, the higher The
        change of sampling this transition will be. Scaling is Softmax
        exponential scaling.

        Arguments:
            batch_size {batch_size} -- the size of the sampled transitions

        Returns:
            old_state, new_state, action, reward, -- Transitions from the replaybuffer
        """
        if not self.filled:
            # compute the weights first
            rewards = self.reward[:self.write_idx]
            rewards_shifted = rewards + np.min(rewards)
            exp_rewards = rewards_shifted**self.alpha
            sum_of_entries = np.sum(exp_rewards)
            weights = exp_rewards / sum_of_entries
            indices = np.random.choice(self.write_idx, size=batch_size, p=weights)
        else:
            # compute the weights first
            rewards = self.reward
            rewards_shifted = rewards + np.min(rewards)
            exp_rewards = rewards_shifted**self.alpha
            sum_of_entries = np.sum(exp_rewards)
            weights = exp_rewards / sum_of_entries
            indices = np.random.choice(self.max_buffer_size, size=batch_size, p=weights)
        old_state = self.old_state[indices]
        new_state = self.new_state[indices]
        action = self.action[indices]
        reward = self.reward[indices]

        return old_state, new_state, action, reward

    def adjust_rewards(self, additional_reward):
        # todo: validate the heck out of this
        # very important!!!
        add_len = len(additional_reward)
        if self.write_idx - add_len < 0:
            remainder = add_len-self.write_idx
            # first fill the rear end of the buffer
            self.reward[self.max_buffer_size-remainder:]\
                += additional_reward[:remainder]
            # now fill the new front end of the buffer
            self.reward[:self.write_idx] = additional_reward[remainder:]
        else:
            self.reward[self.write_idx-add_len:self.write_idx] += additional_reward

    def reset(self):
        """calls both indx_reset functions to fully reset the buffer"""
        self._reset_read_idx()
        self._reset_write_idx()

    def _isempty(self):
        return self.read_idx == self.write_idx

    def _isfull(self):
        return self.write_idx == self.max_buffer_size

    def _reset_read_idx(self):
        """resets the read_idx to 0"""
        self.read_idx = 0

    def _reset_write_idx(self):
        """resets the write_idx to 0"""
        self.write_idx = 0

    def __repr__(self):
        pass
