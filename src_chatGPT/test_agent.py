import unittest

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from agent_chatGPT import DQNAgent

m = 5  # number of cities
t = 24  # number of hours
d = 7  # number of days


class TestDQNAgent(unittest.TestCase):
	def setUp(self):
		self.state_size = m + t + d
		self.action_size = 5
		self.agent = DQNAgent(self.state_size, self.action_size)

	def test_build_model(self):
		model = self.agent.build_model()
		self.assertIsInstance(model, Sequential)
		self.assertEqual(model.layers[0].input_shape, (None, self.state_size))
		self.assertEqual(model.layers[1].output_shape, (None, 24))
		self.assertEqual(
			model.layers[2].output_shape, (None, self.action_size))

	def test_get_action(self):
		state = (1, 2, 3)
		possible_actions_index = [1, 2, 3, 4]
		action = self.agent.get_action(state, possible_actions_index)
		self.assertIn(action, possible_actions_index)

	def test_append_sample(self):
		state = [0, 0, 0]
		action = 0
		reward = 1
		next_state = [1, 0, 0]
		done = False
		agent = DQNAgent(self.state_size, self.action_size)
		agent.append_sample(state, action, reward, next_state, done)
		self.assertEqual(len(agent.memory), 1)
		self.assertEqual(agent.memory[0][0], state)
		self.assertEqual(agent.memory[0][1], action)
		self.assertEqual(agent.memory[0][2], reward)
		self.assertEqual(agent.memory[0][3], next_state)
		self.assertEqual(agent.memory[0][4], done)

	def test_convert_state_to_vector(self):
		agent = DQNAgent(state_size=m+t+d, action_size=5)
		state = (1, 2, 3)
		state_encod = agent.convert_state_to_vector(state)
		assert state_encod[1] == 1
		assert state_encod[m+2] == 1
		assert state_encod[m+t+3] == 1
		assert state_encod[0] == 0
		assert state_encod[m+t] == 0

	def test_get_action(self):
		agent = DQNAgent(state_size=m+t+d, action_size=5)
		state = (1, 2, 3)
		possible_actions_index = [0, 1, 2, 3, 4]
		action = agent.get_action(state, possible_actions_index)
		assert action in possible_actions_index

	def test_train_model(self):
		agent = DQNAgent(state_size=m+t+d, action_size=5)
		state = (1, 2, 3)
		next_state = (4, 5, 6)
		possible_actions_index = [0, 1, 2, 3, 4]
		agent.append_sample(state, 1, 2, next_state, 0)
		agent.append_sample(state, 2, 3, next_state, 1)
		agent.train_model()
		assert agent.memory
		assert agent.model

if __name__ == "__main__":
    agent = DQNAgent(state_size=m + t + d, action_size=5)
    unittest.main()
