import pickle
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
num_devices = len(tf.config.list_physical_devices())

class DQNAgent:
	def __init__(self, state_size, action_size):
		"""
		Initialize the DuelingQAgent class.

		Parameters:
		state_size (int): The size of the state space.
		action_size (int): The size of the action space.
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.discount_factor = 0.95
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_decay = -0.00045
		self.epsilon_min = 0.0000001

		self.batch_size = 4096
		self.memory = deque(maxlen=2000)

		self.model = self.build_model()
		self.initialize_track_state()

	
	def build_model(self):
		"""Build the neural network model for the DQN"""
		if num_devices>2:
			strategy = tf.distribute.MirroredStrategy()
			with strategy.scope():
				model = Sequential()
				model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.state_size, 1)))
				model.add(MaxPooling1D(pool_size=2))
				model.add(Conv1D(64, kernel_size=3, activation='relu'))
				model.add(MaxPooling1D(pool_size=2))
				model.add(Flatten())
				model.add(Dense(64, activation='relu'))
				model.add(Dense(self.action_size, activation='linear'))
				model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
		else:
			model = Sequential()
			model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.state_size, 1)))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Conv1D(64, kernel_size=3, activation='relu'))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Flatten())
			model.add(Dense(64, activation='relu'))
			model.add(Dense(self.action_size, activation='linear'))
			model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
		return model


	def inverse_time_decay(self, initial_epsilon, decay_rate, step):
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
	
	def exponential_decay(self, step: int) -> float:
		"""
		This function returns the epsilon value for the given step using exponential decay.
		The epsilon value starts at the initial_epsilon and decays exponentially with decay_rate.
		It is used as a part of exploration-exploitation trade-off of reinforcement learning.
		"""
		return self.epsilon_min + (
			self.epsilon_max - self.epsilon_min
		) * np.exp(self.epsilon_decay * step)

	def get_action(self, state, possible_actions_index):
		"""
		Get the next action to take using epsilon-greedy policy and decay in epsilon after each sample from the environment.

		Parameters:
		state (array-like): The current state of the agent.
		possible_actions_index (list of int): The possible actions that can be taken in the current state.

		Returns:
		int: the next action that should be taken by
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
		"""
		Save the model's weights to a file.

		Parameters:
		name (str): The name of the file to save the weights to.
		"""
		self.model.save(name)

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
