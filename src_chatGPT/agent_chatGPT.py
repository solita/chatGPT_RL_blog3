import pickle
import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Hyperparameters for the DQN
        self.discount_factor = 0.95
        self.learning_rate = 0.01
        # Added a hyperparameter for epsilon
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01

        self.batch_size = 32
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.initialize_track_state()

    def build_model(self):
        """Build the neural network model for the DQN"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary
        return model

    def get_action(self, state, possible_actions_index):
        """get action from model using epsilon-greedy policy"""
        """
        I transferred the epsilon decay method to the notebook.
        The chatGPT generated function is only choosing a random action or the action with the highest predicted Q-value.
        It should also be considering the possible actions that are available in the current state. Additionally, the function is only decreasing epsilon after each episode, while it should be decreasing epsilon after each sample.
        I don't want to pass the environment class as a parameter to access the env.requests() function. We'll just pass the possible action indices and actions an rewrite this function.
		"""
        if np.random.rand() <= self.epsilon:
            # explore: choose a random action from possible actions
            return random.choice(possible_actions_index)
        else:
            # exploit: choose the action with the highest predicted Q-value
            state = np.array(self.convert_state_to_vector(state)
                             ).reshape(1, self.state_size)
            q_vals = self.model.predict(state, verbose=0)
            return possible_actions_index[np.argmax(q_vals[0][possible_actions_index])]

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def convert_state_to_vector(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros((m + t + d))
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
        return state_encod

    def train_model(self):
        """
        Function to train the model on each step run.
        Picks the random memory events according to batch size and
        runs it through the network to train it.
        """
        """
		This boilerplate from chatGPT won't quite do. It is updating the Q values for one sample at a time,
		and not using a batch sampled from the memory. Using a batch will speed up training and stabilize the model.
		"""
        # 1. Update your 'update_output' and 'update_input' batch
        # 2. Predict the target from earlier model
        # 3. Get the target for the Q-network
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from the memory
        mini_batch = random.sample(self.memory, self.batch_size)
        # Use list comprehension to extract states, actions, rewards, next_states, and done_boolean from mini_batch
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Use numpy operations to convert states and next_states to vectors
        update_input = np.array([self.convert_state_to_vector(state) for state in states])
        update_output = np.array([self.convert_state_to_vector(next_state) for next_state in next_states])

        target = self.model.predict(update_input, verbose=0)
        target_qval = self.model.predict(update_output, verbose=0)

        # update target values
        for i in range(self.batch_size):
            if dones[i]:  # terminal state
                target[i][actions[i]] = rewards[i]
            else:  # non-terminal state
                target[i][actions[i]] = rewards[i] + \
                    self.discount_factor * np.max(target_qval[i])

        # model fit
        self.model.fit(update_input, target,
                       batch_size=self.batch_size, epochs=1, verbose=0)

    def save(self, name):
        self.model.save(name)

    # These functions were added by Jokke Ruokolainen for solution comparison and convenience

    def initialize_track_state(self):
        self.states_tracked_1 = []
        self.states_tracked_2 = []
        self.track_state_1 = self.create_track_state((2, 4, 6), 11)
        self.track_state_2 = self.create_track_state((1, 2, 3), 6)
        self.track_sample_state(self.states_tracked_1, self.track_state_1, 11)
        self.track_sample_state(self.states_tracked_2, self.track_state_2, 6)

    def create_track_state(self, state_values, action_index):
        state = np.array(self.convert_state_to_vector(
            state_values)).reshape(1, self.state_size)
        return state, action_index

    def track_sample_state(self, states_tracked, track_state, action_index):
        states_tracked.append((track_state, action_index))

    def save_tracking_states(self):
        # Use the model to predict the q_value of the state we are tracking.
        q_values = self.model.predict(np.concatenate(
            [self.track_state_1, self.track_state_2]), verbose=0)
        # action (2,3) at index 11 in the action space
        self.states_tracked_1.append(q_values[0][11])
        # action (1,2) at index 6 in the action space
        self.states_tracked_2.append(q_values[1][6])

    def save_weights_numpy(self, name):
        weights = self.model.get_weights()
        try:
            with open(name, "wb") as fpkl:
                pickle.dump(weights, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
