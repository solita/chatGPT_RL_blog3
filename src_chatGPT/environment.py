import random

import numpy as np

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

	def __init__(self):
		"""
    	Initializes the state, action space, and state space for the taxi environment.
    
    	The function creates a list of all possible actions, excluding riding from a location to itself.
    	It also defines the state space as all possible combinations of location, hour, and day.
    	The initial state is chosen randomly from the state space.
    	"""
		# create a list of all possible actions, excluding riding from a location to itself
		self.action_space = [[p, q] for p in range(
			m) for q in range(m) if p != q or p == 0]
		# all possible pickup and drop-off locations, plus the option to go offline
		# insert [0,0] at the beginning of the action space for no ride action
		self.action_space.insert(0, [0, 0])
		self.state_space = [[i, j, k] for i in range(m) for j in range(
			t) for k in range(d)]  # all possible combinations of location, hour, and day
		self.state_init = random.choice(self.state_space)

		# Start the first round
		self.reset()

	def requests(self, state):
		"""
		Determining the number of requests basis the location. 
		Use the table specified in the MDP and complete for rest of the locations
		"""
		location = state[0]
		requests_map = [2, 12, 4, 7, 8]
		requests = min(np.random.poisson(requests_map[location]), 15)

		# (0,0) implies no action. The driver is free to refuse customer request
		# at any point in time.
		# Hence, add the index of action (0,0)->[0] to account for no ride action
		possible_actions_index = [0] + \
			random.sample(range(1, (m - 1) * m + 1), requests)

		actions = [self.action_space[i] for i in possible_actions_index]
		return possible_actions_index, actions

	def calculate_time(self, offline, current_state, pickup, dropoff, current_h, current_d, Time_matrix):
		"""
    	Calculates the total ride time from the current state to the next state, given the action.
    	
		Params:
    	offline: a boolean value indicating if the action is to go offline or not
    	current_state: the current location of the taxi
    	pickup: the pickup location of the next ride
    	dropoff: the dropoff location of the next ride
    	current_h: the current hour
    	current_d: the current day
    	Time_matrix: a 4D matrix containing the time it takes to travel between different locations at different times

    	Returns:
    	The total ride time as an integer
    	"""
		# if offline action then move time by 1
		wait_time = 1 if offline else 0
		transit_time = Time_matrix[current_state][pickup][current_h][current_d] if pickup != current_state else 0
		ride_time = Time_matrix[pickup][dropoff][current_h][current_d] if pickup != current_state else Time_matrix[current_state][dropoff][current_h][current_d]
		return wait_time + transit_time + ride_time
	
	def reward_func(self, state, action, Time_matrix):
		"""
    	Calculates the reward for the given state, action, and Time_matrix.
    	
		Params:
    	state: a tuple representing the current state of the taxi, containing the current location, hour, and day
    	action: a list of two integers representing the pickup and dropoff locations
    	Time_matrix: a 4D matrix containing the time it takes to travel between different locations at different times
    
    	Returns:
    	The reward as a float
    	"""
		pickup, dropoff = action
		current_state, current_h, current_d = state
		offline = True if action == [0, 0] else False
		total_ride_time = self.calculate_time(offline, current_state, pickup, dropoff, current_h, current_d, Time_matrix)
		reward = (total_ride_time * R) - (total_ride_time * C)
		return reward

	def next_state_func(self, state, action, Time_matrix):
		"""
		Takes current state, action and Time_matrix as input and returns the next state and the total ride time.
		
		Params:
    	state: a tuple representing the current state of the taxi, containing the current location, hour, and day
    	action: a list of two integers representing the pickup and dropoff locations
    	Time_matrix: a 4D matrix containing the time it takes to travel between different locations at different times
		
		Returns:
		The next state as list
		"""
		if action == [0, 0]:
			return state, 1
		else:
			pickup, dropoff = action
			current_state, current_h, current_d = state
			offline = True if action == [0, 0] else False
			total_ride_time = self.calculate_time(offline, current_state, pickup, dropoff, current_h, current_d, Time_matrix)
			hour = (current_h + total_ride_time) % t
			day = (current_d + (current_h + total_ride_time) // t) % d
			return [dropoff, int(hour), int(day)], int(total_ride_time)

	def step(self, state, action, Time_matrix):
		"""
    	Takes the current state, action, and Time_matrix as input and returns the next state, reward, and step_time.
    	The current state is represented as a tuple containing the current location, hour, and day.
    	The action is represented as a list of two integers representing the pickup and dropoff locations.
    	The Time_matrix is used to calculate the next state and the reward.
    
    	Params:
        state: a tuple representing the current state of the taxi, containing the current location, hour, and day
        action: a list of two integers representing the pickup and dropoff locations
        Time_matrix: a 4D matrix containing the time it takes to travel between different locations at different times
        
    	Returns:
        next_state: a tuple representing the next state of the taxi, containing the next location, hour, and day
        reward: a float representing the reward for the given state and action
        step_time: an int representing the total time taken for the step
    	"""
		next_state, step_time = self.next_state_func(state, action, Time_matrix)
		reward = self.reward_func(state, action, Time_matrix)
		return next_state, reward, step_time
		
	def reset(self):
		"""
		Resets the environment and returns the action space, state space, and initial state.

		Returns:
    	action_space: the action space of the environment
    	state_space: the state space of the environment
    	state_init: the initial state of the environment
		"""
		return self.action_space, self.state_space, self.state_init
