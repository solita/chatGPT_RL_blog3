# Import routines

import math
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
		"""initialise your state and define your action space and state space"""
		"""
			The init for action_space that chatGPT generated was wrong since it generates a list of all possible pairs of integers between 0 and m-1,
			including pairs where the integers are the same, and an additional pair [0, 0].
			You cant travel from state 0 to 0 in our constrained environment. Instead the time aspect is handled in 
			get_next_state function. Added a fix so that the action_space
			code generates a list of all possible pairs of integers between 0 and m-1,
			where the integers in each pair are distinct and at least one of them is 0.
		"""
		# create a list of all possible actions, excluding riding from a location to itself
		self.action_space = [[p, q] for p in range(
			m) for q in range(m) if p != q or p == 0]
		# all possible pickup and drop-off locations, plus the option to go offline
		# insert [0,0] at the beginning of the action space for no ride action
		self.action_space.insert(0, [0, 0])
		"""
		1.The chatGPT input each state as a tuple, not a bad choice SINCE the location, hour, day is looked up 
		from the Time_matrix so they don't change in our setting. But for sake of clarity
		I'll change this to list since that was my original solution.
		"""
		self.state_space = [[i, j, k] for i in range(m) for j in range(
			t) for k in range(d)]  # all possible combinations of location, hour, and day
		self.state_init = random.choice(self.state_space)

		# Start the first round
		self.reset()

	# Encoding state (or state-action) for NN input

	#def convert_state_to_vector(self, state):
	#		"""convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
	#	state_encod = np.zeros((m + t + d))
	#	state_encod[state[0]] = 1
	#	state_encod[m + state[1]] = 1
	#	state_encod[m + t + state[2]] = 1
	#	return state_encod

	# Getting number of requests

	def requests(self, state):
		"""
		Determining the number of requests basis the location. 
		Use the table specified in the MDP and complete for rest of the locations
		"""
		location = state[0]
		"""
		chatGPT only handled location 0, added handling for the rest.
		Using dictionary instead of if-else suggested by chatGPT.
		And it does not add the index [0] to indicate no ride action the method 
		it suggested just return and empty list
		"""
		requests_map = {0: 2, 1: 12, 2: 4, 3: 7, 4: 8}
		requests = min(np.random.poisson(requests_map[location]), 15)

		# (0,0) implies no action. The driver is free to refuse customer request
		# at any point in time.
		# Hence, add the index of action (0,0)->[0] to account for no ride action
		"""
		chatGPTs implementation of possible_actions_indexes resulted in an assertation error
		assertEqual(len(possible_actions_index), len(actions))
			AssertionError: 3 != 4
		This was due to the fact that it appended the offline action a second time here, that was already added to the action_space in init.
		Hence it was already in the actions list since the driver always has the option to go offline.
		"""
		possible_actions_index = [0] + \
			random.sample(range(1, (m - 1) * m + 1), requests)

		actions = [self.action_space[i] for i in possible_actions_index]
		return possible_actions_index, actions

	def reward_func(self, state, action, Time_matrix):
		"""Takes in state, action and Time-matrix and returns the reward"""
		"""
			1. No-ride action is not handled correctly, it should move the time component 1h as described in README.md
			the function was returning the reward -C which does not correspond to the reward calculation formula: (time * R) - (time * C)
		"""
		pickup, dropoff = action
		current_state, current_h, current_d = state
		if action == [0, 0]:
			total_ride_time = 1
			reward = (total_ride_time * R) - (total_ride_time * C)
		else:
			"""
			1. No-ride action is not handled correctly. No-ride action should move the time component +1h as described in problem definition. The function was returning the reward=-C which does not correspond to the reward calculation formula: (time * R) - (time * C). time = total transit time from current location through pickup to dropoff (transitioning from current state to next state).
			2.chatGPT is calculating the travel time from A>B and updating the location. chatGPT is doing a mistake, hour and day in a state tuple are integers.
				chatGPTs way of calculating the time it takes to transition (for the taxi to drive) from current state to next state results in returning arrays for h and d. 
				This is due to the fact that chatGPT is slicing the 4D Time-Matrix in a wrong manner. ChatGPT is using two sets of indices, pickup and dropoff, to slice the array.
				4. indices are needed to slice the array in the correct way. I broke the total transition time calculation to multiple steps for clarity

			"""
			# if offline action then move time by 1
			wait_time = 1 if action == [0, 0] else 0
			# ride time from current state (location) to next pickup location
			transit_time = Time_matrix[current_state][pickup
                                             ][current_h][current_d] if pickup != current_state else 0
			# ride time from the pickup location to next dropoff (next state)
			ride_time = Time_matrix[pickup][dropoff][current_h][current_d
                                                                ] if pickup != current_state else Time_matrix[current_state][dropoff][current_h][current_d]
			total_ride_time = wait_time + transit_time + ride_time
			reward = (total_ride_time * R) - (total_ride_time * C)
		return reward

	def next_state_func(self, state, action, Time_matrix):
		"""Takes state and action as input and returns next state"""
		"""
			1. The if else condition is wrong here. If action is to go-offline [0,0] then the total_ride_time is moved forward by 1
		"""
		if action == [0, 0]:
			return state, 1
		else:
			"""
			3. Here chatGPT is calculating the travel time from A>B and updating the location.
			chatGPT is doing a mistake, hour and day in a state tuple are integers.
			This way of calculating the time from A to B results in returning arrays for h and d
			this is due to the fact that chatGPT is slicing the 4D TimeMatrix in a wrong manner.
			chatGPT is using two sets of indices pickup and dropoff to slice the array.
			4 indices are actually needed to slice the array in a correct way. I'll break the time
			calculation to multiple steps for clarity
			"""
			pickup, dropoff = action
			current_state, current_h, current_d = state
			# if offline action then move time by 1
			wait_time = 1 if action == [0, 0] else 0
			# ride time from current state (location) to next pickup location
			transit_time = Time_matrix[current_state][pickup
                                             ][current_h][current_d] if pickup != current_state else 0
			# ride time from the pickup location to next dropoff (next state)
			ride_time = Time_matrix[pickup][dropoff][current_h][current_d
                                                                ] if pickup != current_state else Time_matrix[current_state][dropoff][current_h][current_d]
			total_ride_time = wait_time + transit_time + ride_time
			# update the hour and day for next_state
			hour = (current_h + total_ride_time) % t
			day = (current_d + (current_h + total_ride_time) // t) % d
			# chatGPT returned a tuple I decided to use a list
			return [dropoff, int(hour), int(day)], int(total_ride_time)

	def step(self, state, action, Time_matrix):
			next_state, step_time = self.next_state_func(state, action, Time_matrix)
			reward = self.reward_func(state, action, Time_matrix)
			return next_state, reward, step_time
		
	def reset(self):
		return self.action_space, self.state_space, self.state_init
