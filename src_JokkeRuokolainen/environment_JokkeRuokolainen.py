import random

import jax.numpy as jnp
import numpy as np

# Defining hyperparameters
m = 5  # number of cities
t = 24  # number of hours
d = 7  # number of days
C = 5  # Per hour battery consumption
R = 9  # Per hour revenue from a passenger


class CabDriver:
    def __init__(self):
        """ Iitializes your state and defines your action space and state space

        Fixes made:
                1.imports not needed math library removed.
                2.Instead of using np.random.choice(len(self.state_space)) to select a random state, we can use np.random.choice(self.state_space) this is faster and more readable.

        Parameters:

        state_size (int): The size of the state space.
        action_size (int): The size of the action space.
        """
        # create a list of all possible actions, excluding riding from a city to itself
        self.action_space = [[p, q] for p in range(
            m) for q in range(m) if p != q or p == 0]
        # insert [0,0] at the beginning of the action space for no ride action
        self.action_space.insert(0, [0, 0])

        # create a list of all possible states, represented as a combination of location, time, and day
        self.state_space = [[loc, time, day] for loc in range(
            m) for time in range(t) for day in range(d)]

        # randomly initialize the current state
        self.state_init = self.state_space[np.random.choice(
            len(self.state_space))]

        # set total drive time to zero
        self.total_time = 0

        # start the first round
        self.reset()

    def requests(self, state):
        """Determining the number of requests based on the location
        This method implements a replay buffer for increase stability during DQN training
        Fixes made:
                1.Instead of using if-else statements, use a dictionary to map the location to the number of requests, this is more efficient and readable.
                2.Use min(np.random.poisson(requests_map[location]), 15) to limit the requests to 15.
                3.Use list comprehension to append the [0] to the list of random actions instead of list concatenation which is more efficient.
        """
        location = state[0]
        # Use a dictionary to map the location to the number of requests
        # Instead of using if-else statements
        requests_map = {0: 2, 1: 12, 2: 4, 3: 7, 4: 8}
        requests = min(np.random.poisson(requests_map[location]), 15)

        # use list comprehension instead of using list concatenation
        possible_actions_index = [0] + \
            random.sample(range(1, (m - 1) * m + 1), requests)

        actions = [self.action_space[i] for i in possible_actions_index]
        return possible_actions_index, actions

    def update_time_day(self, curr_time, curr_day, ride_duration):
        """Takes in the current time, current day and duration taken for driver's journey and returns
        updated time and updated day post that journey.
        Fixes Made:
                1. instead of having two separate if-else for updating time and day, combine both of them into two lines of code.
                        as the operation of updated_day = (curr_day + num_days) % 7 is included inside the if-else block, this is moved outside of it.
                2. updated_day = curr_day statement removed as it is not necessary
        """
        ride_duration = int(ride_duration)
        updated_time = (curr_time + ride_duration) % 24
        updated_day = (curr_day + (curr_time + ride_duration) // 24) % 7
        return updated_time, updated_day

    def get_next_state_and_time_func(self, state, action, Time_matrix):
        """Takes state, action and Time_matrix as input and returns next state, wait_time, transit_time, ride_time.
        Fixes Made:
                1.Instead of using the nested if-else block to calculate the wait_time, transit_time, and ride_time, we can use ternary operator to simplify and make the code more readable.
                2.The next_loc is equal to the drop_loc, so it is not necessary to recalculate it at the end.
                3.This method returns all times in the same unit, it would be more clear if the function mention the unit of time
        """
        # Initialize various times
        wait_time = 1 if action == [0, 0] else 0
        transit_time = Time_matrix[state[0]][action[0]
                                             ][state[1]][state[2]] if action[0] != state[0] else 0
        ride_time = Time_matrix[action[0]][action[1]][state[1]][state[2]
                                                                ] if action[0] != state[0] else Time_matrix[state[0]][action[1]][state[1]][state[2]]
        self.total_time = wait_time + transit_time + ride_time

        # update time and day, and set next_loc
        next_time, next_day = self.update_time_day(
            state[1], state[2], self.total_time)
        next_loc = action[1]

        # Finding next_state using the next_loc and the next time states.
        next_state = [next_loc, next_time, next_day]

        return next_state, wait_time, transit_time, ride_time

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time_matrix and returns the reward
        """
        _, wait_time, transit_time, ride_time = self.get_next_state_and_time_func(
            state, action, Time_matrix)
        idle_time = wait_time + transit_time
        customer_ride_time = ride_time
        return (R * customer_ride_time) - (C * (customer_ride_time + idle_time)) if customer_ride_time != 0 else -C

    def step(self, state, action, Time_matrix):
        """Take a trip as a cab driver. Takes state, action and Time_matrix as input and returns next_state, reward and total time spent
        Fixes Made:
                1.in the function, the state_tuple is not necessary, the function only use the value that the tuple contain
                2.the output of the function get_next_state_and_time_func can be unpack directly to multiple variable
        """
        # Get the next state and the various time durations
        next_state, wait_time, transit_time, ride_time = self.get_next_state_and_time_func(
            state, action, Time_matrix)

        # Calculate the reward and total_time of the step
        reward = self.reward_func(state, action, Time_matrix)
        total_time = wait_time + transit_time + ride_time

        return next_state, reward, total_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
