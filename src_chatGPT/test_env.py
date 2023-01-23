import unittest

import numpy as np
from environment import CabDriver

# Defining hyperparameters
m = 5  # number of cities
t = 24  # number of hours
d = 7  # number of days
C = 5  # Per hour battery consumption
R = 9  # Per hour revenue from a passenger
max_time = 24 * 30

class TestCabDriver(unittest.TestCase):
    def setUp(self):
        self.driver = CabDriver()

    def test_init(self):
        self.assertEqual(len(self.driver.action_space), m*(m-1) + 2) # +2 so that we take into account the go offline actoin that was appended to the action_space
        self.assertEqual(len(self.driver.state_space), m*t*d)
        self.assertIsInstance(self.driver.state_init, list)
        self.assertNotEqual(self.driver.state_init, [0,0,0])
        self.assertIn(self.driver.state_init, self.driver.state_space)       
        
    def test_requests(self):
        state = [0, 0, 0]
        possible_actions_index, actions = self.driver.requests(state)
        self.assertIsInstance(possible_actions_index, list)
        self.assertIsInstance(actions, list)
        self.assertEqual(len(possible_actions_index), len(actions))
        self.assertIn(0, possible_actions_index)
        for index, action in zip(possible_actions_index, actions):
            self.assertIn(index, range(len(self.driver.action_space)))
            self.assertIn(action, self.driver.action_space)
            # This tests that the driver can't drive a loop from 1 to 1 
            # but there is the exception of going offline which is (0,0)
            if action[0] == 0 and action[1] == 0:
                pass
            else:
                self.assertNotEqual(action[0], action[1])

if __name__ == '__main__':
    unittest.main()


