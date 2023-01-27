import pickle
import random
from collections import deque

import jax.numpy as jnp
import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.losses import huber
from keras.optimizers import Adam

# from .prb import PrioritizedReplayBuffer
# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour battery consumption
R = 9  # Per hour revenue from a passenger
MAX_TIME = 24 * 30
buffer_size = 2000


class DuelingQAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize the DuelingQAgent class.

        Parameters:
        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.95  # gamma
        self.epsilon = 1.0  # exploration rate
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01
        self.learning_rate = 0.01

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = 64

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build the Dueling Q-Network model for the agent.
        """
        input_state = Input(shape=(self.state_size,))
        x = Dense(64, activation='swish')(input_state)
        x = Dense(64, activation='swish')(x)
        value = Dense(1, activation='linear')(x)
        advantage = Dense(self.action_size, activation='linear')(x)
        output = Lambda(
            lambda x: x[0] + (x[1]-K.mean(x[1], axis=1, keepdims=True)))([value, advantage])
        model = Model(inputs=input_state, outputs=output)
        model.compile(loss=huber(), optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def exponential_decay(self, initial_epsilon: float, decay_rate: float, step: int) -> float:
        """
        This function returns the epsilon value for the given step using exponential decay.
        The epsilon value starts at the initial_epsilon and decays exponentially with decay_rate.
        It is used as a part of exploration-exploitation trade-off of reinforcement learning.

        Parameters:
        initial_epsilon (float): The starting value of epsilon.
        decay_rate (float): The rate at which epsilon decays.
        step (int): The current step of the training.

        Returns:
        float: The epsilon value for the given step. 
        """
        return initial_epsilon * np.exp(-decay_rate * step)

    def inverse_time_decay(self, initial_epsilon: float, decay_rate: float, step: int) -> float:
        """
        This function returns the epsilon value for the given step using inverse time decay.
        The epsilon value starts at the initial_epsilon and decays inversely proportional to the step number with decay_rate.
        It is used as a part of exploration-exploitation trade-off of reinforcement learning.

        Parameters:
        initial_epsilon (float): The starting value of epsilon.
        decay_rate (float): The rate at which epsilon decays.
        step (int): The current step of the training.

        Returns:
        float: The epsilon value for the given step.
        """
        return initial_epsilon / (1 + decay_rate * step)

    def update_target_model(self):
        """
        Update the target model with the weights of the model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def append_sample(self, state, action, reward, next_state, done):
        """
        Add a new experience to the agent's memory.

        Parameters:
        state (array-like): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received for taking the action in the current state.
        next_state (array-like): The state reached after taking the action in the current state.
        done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, possible_actions_index):
        """
        Get the next action to take using epsilon-greedy policy and decay in epsilon after each sample from the environment.

        Parameters:
        state (array-like): The current state of the agent.
        possible_actions_index (list of int): The possible actions that can be taken in the current state.

        Returns:
        int: the next action that should be taken by
        """
        # Get a list of the ride requests driver got.
        # get action from model using epsilon-greedy policy
        # variable z randomly chooses a value between [0,1)
        z = np.random.random()
        if z <= self.epsilon:
            # explore: choose a random action from all possible actions
            return random.choice(possible_actions_index)
        else:
            # choose the action with the highest q(s, a)
            state = np.array(self.convert_state_to_vector(
                state)).reshape(1, self.state_size)
            q_vals = self.model.predict(state, verbose=0)
            return possible_actions_index[np.argmax(q_vals[0][possible_actions_index])]

    def convert_state_to_vector(self, state):
        """
        Convert the input state to a vector representation.

        Parameters:
        state (array-like): The current state of the agent.

        Returns:
        list of float : the vector representation of the input state
        """
        # create a zero vector of size m + t + d
        state_encod = jnp.zeros(m + t + d)
        # set the value at the index corresponding to the location to 1
        state_encod = state_encod.at[state[0]].set(1)
        # set the value at the index corresponding to the time to 1
        state_encod = state_encod.at[m + int(state[1])].set(1)
        # set the value at the index corresponding to the day to 1
        state_encod = state_encod.at[m + t + int(state[2])].set(1)

        return state_encod

    # pick samples randomly from replay memory (with batch_size) and train the network
    def train_model(self):
        """
        Train the model by picking random memories from the memory buffer and running them through the network.
        """
        if len(self.memory) > self.batch_size:
            # Sample batch from the memory
            mini_batch = random.sample(self.memory, self.batch_size)
            # Use list comprehension to extract states, actions, rewards, next_states, and done_boolean from mini_batch
            states, actions, rewards, next_states, dones = zip(
                *mini_batch)

            # Use numpy operations to convert states and next_states to vectors
            update_input = np.array(
                [self.convert_state_to_vector(state) for state in states])
            update_output = np.array([self.convert_state_to_vector(
                next_state) for next_state in next_states])

            # predict the target q-values from states s
            target = self.model.predict(update_input, verbose=0)

            # update the target values
            for i in range(self.batch_size):
                if dones[i]:  # terminal state
                    target[i][actions[i]] = rewards[i]
                else:  # non-terminal state
                    # target for q-network
                    target_qval = self.target_model.predict(
                        update_output, verbose=0)
                    target[i][actions[i]] = rewards[i] + self.discount_factor * np.max(
                        target_qval[i]
                    )
            # 4. Fit your model and track the loss values
            # model fit
            self.model.fit(
                update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
            )

    def load(self, name):
        """
        Load the model's weights from a file.

        Parameters:
        name (str): The name of the file to load the weights from. 
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Save the model's weights to a file.

        Parameters:
        name (str): The name of the file to save the weights to.
        """
        self.model.save_weights(name)

# These functions were added by Jokke Ruokolainen for solution comparison and convenience

    def initialize_track_state(self):
        """
        Initialize the state tracking variables and sets the initial state to track.
        """
        self.states_tracked_1 = []
        self.states_tracked_2 = []
        self.track_state_1 = self.create_track_state((2, 4, 6), 11)
        self.track_state_2 = self.create_track_state((1, 2, 3), 6)
        self.track_sample_state(self.states_tracked_1, self.track_state_1, 11)
        self.track_sample_state(self.states_tracked_2, self.track_state_2, 6)

    def create_track_state(self, state_values, action_index):
        """
        Create a tuple of the state and its corresponding action index to track.

        Parameters:
        state_values (tuple of int): The values of the state to track.
        action_index (int): The index of the action corresponding to the state. 

        Returns:
        tuple : A tuple containing the state and its corresponding action index.
        """
        state = np.array(self.convert_state_to_vector(
            state_values)).reshape(1, self.state_size)
        return state, action_index

    def track_sample_state(self, states_tracked, track_state, action_index):
        """
        Add the state and its corresponding action index to the list of states to be tracked.

        Parameters:
        states_tracked (list): A list of states and their corresponding action indices to track.
        track_state (tuple): The state and its corresponding action index to be added to the list.
        action_index (int): The index of the action corresponding to the state.
        """
        states_tracked.append((track_state, action_index))

    def save_tracking_states(self):
        """
        Save the q-values of the tracked states using the model's predictions.
        """
        q_values = self.model.predict(np.concatenate(
            [self.track_state_1, self.track_state_2]), verbose=0)
        # action (2,3) at index 11 in the action space
        self.states_tracked_1.append(q_values[0][11])
        # action (1,2) at index 6 in the action space
        self.states_tracked_2.append(q_values[1][6])

    def save_weights_numpy(self, name):
        """
        Save the model's weights to a numpy file.

        Parameters:
        name (str): The name of the file to save the weights to.
        """
        weights = self.model.get_weights()
        try:
            with open(name, "wb") as fpkl:
                pickle.dump(weights, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
